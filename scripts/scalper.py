#!/usr/bin/env python3
"""
scalper.py — Autonomous SOL-PERP scalping engine.

Strategy: Detect 0.5%+ price jumps and ride momentum.
Execution: Pure Python via Orderly REST API (no AI in the hot loop).
Journal: Logs trades to journal-cal API.

Architecture:
  - Polls Orderly public market trades every 1-2s for real-time price data
  - Maintains rolling price windows for jump detection
  - Places market orders on detected jumps
  - Manages TP/SL per position
  - Logs everything to journal + local file
"""

import json
import time
import math
import signal
import sys
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

# Add scripts dir to path for orderly_auth
sys.path.insert(0, str(Path(__file__).parent))
from orderly_auth import OrderlyClient

# ── Configuration ─────────────────────────────────────────────────────────

SYMBOL = "PERP_SOL_USDC"
# Load private key from credentials file (never hardcode)
import json as _creds_json
_CREDS_FILE = Path(__file__).resolve().parent.parent / "config" / "orderly_credentials.json"
_creds = _creds_json.loads(_CREDS_FILE.read_text()) if _CREDS_FILE.exists() else {}
SOLANA_KEY = _creds.get("private_key", "")

# Strategy params
JUMP_THRESHOLD_PCT = 0.5       # Min % move to trigger entry
LOOKBACK_SECONDS = 60          # Window to detect jumps
POLL_INTERVAL_SEC = 1          # Price polling interval (was 2s → now 1s)
MAX_LEVERAGE = 10              # Start conservative (account max is 10x, SOL supports 100x)
RISK_PER_TRADE_PCT = 30        # % of balance to risk per trade
TP_MULTIPLIER = 1.5            # Take profit = entry +/- (jump_size * multiplier)
SL_MULTIPLIER = 0.8            # Stop loss = entry -/+ (jump_size * multiplier)
MAX_HOLD_SECONDS = 300         # Force close after 5 min
MIN_NOTIONAL = 10              # Orderly minimum
COOLDOWN_SECONDS = 30          # Min time between trades
MAX_TRADES_PER_HOUR = 5        # Safety limit

# Journal API
JOURNAL_URL = "https://journal-cal.vercel.app"
JOURNAL_PASS = os.environ.get("JOURNAL_PASS", "vest2026")

# Logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ── Data Structures ───────────────────────────────────────────────────────

@dataclass
class PriceTick:
    price: float
    timestamp: float  # unix seconds
    side: str  # BUY/SELL
    qty: float


@dataclass
class Position:
    symbol: str
    side: str  # BUY/SELL
    entry_price: float
    quantity: float
    entry_time: float
    jump_size_pct: float
    tp_price: float
    sl_price: float
    journal_id: Optional[str] = None
    order_id: Optional[int] = None


@dataclass
class TradeStats:
    total: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    trades_this_hour: int = 0
    hour_start: float = field(default_factory=time.time)


# ── Logging Setup ─────────────────────────────────────────────────────────

def setup_logging():
    log_file = LOG_DIR / f"scalper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ]
    )
    return logging.getLogger("scalper")


# ── Journal Integration ───────────────────────────────────────────────────

class JournalClient:
    def __init__(self, base_url=JOURNAL_URL, password=JOURNAL_PASS):
        self.base_url = base_url
        self.session = requests.Session()
        self._login(password)

    def _login(self, password):
        try:
            r = self.session.get(f"{self.base_url}/login?pass={password}", allow_redirects=False)
            if r.status_code in (301, 302):
                # Extract cookie from redirect
                self.session.cookies.update(r.cookies)
            # Follow redirect
            self.session.get(self.base_url)
        except Exception as e:
            logging.warning(f"Journal login failed: {e} — will log locally only")

    def log_entry(self, trade_data: dict) -> Optional[str]:
        try:
            r = self.session.post(f"{self.base_url}/api/trades", json=trade_data)
            if r.status_code == 200:
                return r.json().get("id")
        except Exception as e:
            logging.warning(f"Journal log failed: {e}")
        return None

    def update_trade(self, trade_id: str, patch: dict):
        try:
            self.session.patch(f"{self.base_url}/api/trades/{trade_id}", json=patch)
        except Exception as e:
            logging.warning(f"Journal update failed: {e}")


# ── Scalper Engine ────────────────────────────────────────────────────────

class ScalperEngine:
    def __init__(self, client: OrderlyClient, logger):
        self.client = client
        self.log = logger
        self.journal = None
        self.position: Optional[Position] = None
        self.stats = TradeStats()
        self.price_history: deque = deque(maxlen=500)
        self.last_trade_time = 0
        self.running = False

        # Try to connect to journal
        try:
            self.journal = JournalClient()
            self.log.info("📒 Connected to trade journal")
        except:
            self.log.warning("📒 Journal unavailable — logging locally only")

    def fetch_price(self) -> Optional[PriceTick]:
        """Fetch latest market trade from Orderly."""
        try:
            r = requests.get(
                f"{self.client.base_url}/v1/public/market_trades",
                params={"symbol": SYMBOL, "limit": 1},
                timeout=5
            )
            data = r.json()
            if data.get("success") and data["data"]["rows"]:
                t = data["data"]["rows"][0]
                return PriceTick(
                    price=t["executed_price"],
                    timestamp=t["executed_timestamp"] / 1000.0,
                    side=t["side"],
                    qty=t["executed_quantity"]
                )
        except Exception as e:
            self.log.warning(f"Price fetch error: {e}")
        return None

    def detect_jump(self) -> Optional[tuple]:
        """
        Detect a 0.5%+ price jump in the lookback window.
        Returns (direction, jump_pct, current_price) or None.
        """
        if len(self.price_history) < 5:
            return None

        now = time.time()
        current = self.price_history[-1]

        # Find oldest price within lookback window
        oldest_in_window = None
        for tick in self.price_history:
            if now - tick.timestamp <= LOOKBACK_SECONDS:
                oldest_in_window = tick
                break

        if oldest_in_window is None:
            return None

        # Calculate move
        move_pct = (current.price - oldest_in_window.price) / oldest_in_window.price * 100

        if abs(move_pct) >= JUMP_THRESHOLD_PCT:
            direction = "BUY" if move_pct > 0 else "SELL"
            return (direction, abs(move_pct), current.price)

        return None

    def calculate_position_size(self, price: float) -> float:
        """Calculate position size based on available balance and leverage."""
        try:
            bal = self.client.get_balance()
            holding = 0
            for h in bal.get("data", {}).get("holding", []):
                if h["token"] == "USDC":
                    holding = h["holding"]
                    break

            risk_usd = holding * (RISK_PER_TRADE_PCT / 100)
            notional = risk_usd * MAX_LEVERAGE

            # Ensure min notional
            if notional < MIN_NOTIONAL:
                self.log.warning(f"Notional ${notional:.2f} below min ${MIN_NOTIONAL}. Need more balance.")
                return 0

            # Convert to SOL quantity (round to base_tick = 0.01)
            qty = math.floor(notional / price * 100) / 100
            return max(qty, 0.01)  # base_min

        except Exception as e:
            self.log.error(f"Position size calc error: {e}")
            return 0

    def open_position(self, direction: str, jump_pct: float, price: float):
        """Place a market order and set TP/SL."""
        qty = self.calculate_position_size(price)
        if qty <= 0:
            self.log.warning("⚠️ Cannot open — insufficient balance")
            return

        # Calculate TP and SL
        jump_dollars = price * (jump_pct / 100)
        if direction == "BUY":
            tp = round(price + jump_dollars * TP_MULTIPLIER, 3)
            sl = round(price - jump_dollars * SL_MULTIPLIER, 3)
        else:
            tp = round(price - jump_dollars * TP_MULTIPLIER, 3)
            sl = round(price + jump_dollars * SL_MULTIPLIER, 3)

        self.log.info(f"🎯 OPENING {direction} {qty} SOL @ ~${price:.3f} | TP: ${tp:.3f} SL: ${sl:.3f} | Jump: {jump_pct:.2f}%")

        try:
            result = self.client.create_order(
                symbol=SYMBOL,
                side=direction,
                order_type="MARKET",
                order_quantity=qty,
            )

            if result.get("success"):
                order_id = result["data"]["order_id"]
                self.position = Position(
                    symbol=SYMBOL,
                    side=direction,
                    entry_price=price,
                    quantity=qty,
                    entry_time=time.time(),
                    jump_size_pct=jump_pct,
                    tp_price=tp,
                    sl_price=sl,
                    order_id=order_id,
                )
                self.last_trade_time = time.time()
                self.stats.total += 1
                self.stats.trades_this_hour += 1

                self.log.info(f"✅ Order placed! ID: {order_id}")

                # Log to journal
                if self.journal:
                    jid = self.journal.log_entry({
                        "symbol": "SOL-PERP",
                        "direction": "LONG" if direction == "BUY" else "SHORT",
                        "timeframe": "1m",
                        "strategy": "MomentumScalp",
                        "entry_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "entry_time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                        "entry_price": price,
                        "stop_loss": sl,
                        "target_price": tp,
                        "position_size": qty,
                        "leverage": MAX_LEVERAGE,
                        "risk_amount": qty * price / MAX_LEVERAGE,
                        "thesis": f"Momentum scalp: {jump_pct:.2f}% jump detected in {LOOKBACK_SECONDS}s window",
                        "status": "open",
                        "tags": ["auto", "scalp", "momentum"],
                    })
                    if jid:
                        self.position.journal_id = str(jid)
                        self.log.info(f"📒 Journaled as trade #{jid}")
            else:
                self.log.error(f"❌ Order failed: {result}")

        except Exception as e:
            self.log.error(f"❌ Order error: {e}")

    def check_exit(self, current_price: float):
        """Check if position should be closed (TP/SL/timeout)."""
        if not self.position:
            return

        pos = self.position
        elapsed = time.time() - pos.entry_time
        exit_reason = None
        exit_type = None

        # Check TP
        if pos.side == "BUY" and current_price >= pos.tp_price:
            exit_reason = f"Take profit hit (${pos.tp_price:.3f})"
            exit_type = "tp"
        elif pos.side == "SELL" and current_price <= pos.tp_price:
            exit_reason = f"Take profit hit (${pos.tp_price:.3f})"
            exit_type = "tp"

        # Check SL
        elif pos.side == "BUY" and current_price <= pos.sl_price:
            exit_reason = f"Stop loss hit (${pos.sl_price:.3f})"
            exit_type = "sl"
        elif pos.side == "SELL" and current_price >= pos.sl_price:
            exit_reason = f"Stop loss hit (${pos.sl_price:.3f})"
            exit_type = "sl"

        # Check timeout
        elif elapsed >= MAX_HOLD_SECONDS:
            exit_reason = f"Max hold time ({MAX_HOLD_SECONDS}s)"
            exit_type = "timeout"

        if exit_reason:
            self.close_position(current_price, exit_reason, exit_type)

    def close_position(self, current_price: float, reason: str, exit_type: str):
        """Close current position."""
        if not self.position:
            return

        pos = self.position
        close_side = "SELL" if pos.side == "BUY" else "BUY"

        self.log.info(f"🔄 CLOSING {close_side} {pos.quantity} SOL @ ~${current_price:.3f} | Reason: {reason}")

        try:
            result = self.client.create_order(
                symbol=SYMBOL,
                side=close_side,
                order_type="MARKET",
                order_quantity=pos.quantity,
                reduce_only=True,
            )

            if result.get("success"):
                # Calculate P&L
                if pos.side == "BUY":
                    pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                else:
                    pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100

                pnl_usd = pos.quantity * abs(current_price - pos.entry_price)
                if pnl_pct < 0:
                    pnl_usd = -pnl_usd

                # Update stats
                if pnl_usd > 0:
                    self.stats.wins += 1
                else:
                    self.stats.losses += 1
                self.stats.total_pnl += pnl_usd

                emoji = "🟢" if pnl_usd >= 0 else "🔴"
                self.log.info(f"{emoji} P&L: ${pnl_usd:+.4f} ({pnl_pct:+.3f}%) | Reason: {reason}")
                self.log.info(f"📊 Stats: {self.stats.wins}W/{self.stats.losses}L | Total P&L: ${self.stats.total_pnl:+.4f}")

                # Update journal
                if self.journal and pos.journal_id:
                    error_cat = None
                    if exit_type == "sl":
                        error_cat = "bad_entry"
                    elif exit_type == "timeout":
                        error_cat = "no_follow_through"

                    self.journal.update_trade(pos.journal_id, {
                        "status": "closed",
                        "actual_entry": pos.entry_price,
                        "actual_exit": current_price,
                        "actual_pnl_pct": round(pnl_pct, 3),
                        "actual_pnl_usd": round(pnl_usd, 4),
                        "exit_reason": reason,
                        "exit_type": exit_type,
                        "error_category": error_cat,
                        "lessons": f"Jump: {pos.jump_size_pct:.2f}%, Hold: {time.time()-pos.entry_time:.0f}s",
                    })

                self.position = None
            else:
                self.log.error(f"❌ Close failed: {result}")

        except Exception as e:
            self.log.error(f"❌ Close error: {e}")

    def can_trade(self) -> bool:
        """Check if we're allowed to open a new trade."""
        if self.position:
            return False
        if time.time() - self.last_trade_time < COOLDOWN_SECONDS:
            return False

        # Reset hourly counter
        if time.time() - self.stats.hour_start > 3600:
            self.stats.trades_this_hour = 0
            self.stats.hour_start = time.time()

        if self.stats.trades_this_hour >= MAX_TRADES_PER_HOUR:
            return False

        return True

    def run(self):
        """Main loop."""
        self.running = True
        self.log.info(f"🚀 Scalper starting on {SYMBOL}")
        self.log.info(f"   Jump threshold: {JUMP_THRESHOLD_PCT}% | Lookback: {LOOKBACK_SECONDS}s")
        self.log.info(f"   Leverage: {MAX_LEVERAGE}x | Risk/trade: {RISK_PER_TRADE_PCT}%")
        self.log.info(f"   TP: {TP_MULTIPLIER}x jump | SL: {SL_MULTIPLIER}x jump")
        self.log.info(f"   Max hold: {MAX_HOLD_SECONDS}s | Cooldown: {COOLDOWN_SECONDS}s")
        self.log.info(f"   Max trades/hour: {MAX_TRADES_PER_HOUR}")

        # Check balance
        bal = self.client.get_balance()
        for h in bal.get("data", {}).get("holding", []):
            if h["token"] == "USDC":
                self.log.info(f"   💰 Balance: ${h['holding']:.4f} USDC")

        tick_count = 0
        while self.running:
            try:
                tick = self.fetch_price()
                if tick:
                    self.price_history.append(tick)
                    tick_count += 1

                    # Log every 30 ticks (~1 min)
                    if tick_count % 30 == 0:
                        hi = max(t.price for t in self.price_history)
                        lo = min(t.price for t in self.price_history)
                        spread_pct = (hi - lo) / lo * 100 if lo else 0
                        pos_str = f" | POS: {self.position.side} {self.position.quantity}" if self.position else ""
                        self.log.info(f"📈 SOL ${tick.price:.3f} | Range: ${lo:.3f}-${hi:.3f} ({spread_pct:.2f}%) | Ticks: {len(self.price_history)}{pos_str}")

                    # Check exit on open position
                    if self.position:
                        self.check_exit(tick.price)

                    # Check for new entry
                    elif self.can_trade():
                        jump = self.detect_jump()
                        if jump:
                            direction, jump_pct, price = jump
                            self.log.info(f"⚡ Jump detected: {jump_pct:+.2f}% → {direction}")
                            self.open_position(direction, jump_pct, price)

                time.sleep(1)  # Now polling every 1s (targeting ~15s refresh)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log.error(f"Loop error: {e}")
                time.sleep(5)

        # Cleanup: close any open position
        if self.position:
            tick = self.fetch_price()
            if tick:
                self.close_position(tick.price, "Shutdown", "shutdown")

        self.log.info(f"🛑 Scalper stopped. Final stats: {self.stats.wins}W/{self.stats.losses}L | P&L: ${self.stats.total_pnl:+.4f}")


# ── Entry Point ───────────────────────────────────────────────────────────

def main():
    logger = setup_logging()

    # Connect to Orderly
    logger.info("Connecting to Orderly...")
    client = OrderlyClient(SOLANA_KEY)
    if not client.is_ready:
        logger.info("Running first-time setup...")
        client.setup()

    logger.info(f"✅ Connected as {client.address[:12]}... | Account: {client.account_id[:16]}...")

    # Test trading with micro trade
    logger.info("🧪 Running micro trade test...")
    try:
        client.test_micro_trade(SYMBOL, MIN_NOTIONAL)
        logger.info("✅ Micro trade test passed - trading is working")
    except Exception as e:
        logger.error(f"❌ Micro trade test failed: {e}")
        logger.error("Trading may not work. Check credentials and try again.")
        return

    # Create and run scalper
    engine = ScalperEngine(client, logger)

    # Graceful shutdown
    def handle_signal(sig, frame):
        logger.info("Received shutdown signal...")
        engine.running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    engine.run()


if __name__ == "__main__":
    main()
