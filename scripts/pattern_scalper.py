#!/usr/bin/env python3
"""
pattern_scalper.py — Probabilistic pattern-based scalper for SOL-PERP.

Replaces the simple jump detector with a multi-pattern probabilistic engine.

Entry conditions:
  - composite_confidence > ENTRY_CONFIDENCE_THRESHOLD (0.52)
  - consensus >= MIN_CONSENSUS (0.5)
  - win_rate_weighted > 0.48
  - At least 1 pattern triggered (2+ preferred)

Position sizing scales with confidence:
  base_risk * (0.5 + composite_confidence * 0.5)

On close, calls engine.update_stats() per active pattern to refine win rates.
"""

import json
import math
import signal
import sys
import time
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from orderly_auth import OrderlyClient
from pattern_engine import PatternEngine, CompositeSignal

# ── Configuration ─────────────────────────────────────────────────────────

SYMBOL               = "PERP_SOL_USDC"
SOLANA_KEY           = "5XY4ErjzPekDin7MyBzLcN6Dvd7rn2BRPaGzwZvStpu27uwyp7JXvPYpZfaCJ1nEBMeFoWqginvvfDBERdsKmGUj"

ENTRY_CONFIDENCE_THRESHOLD = 0.52
MIN_CONSENSUS              = 0.5
MIN_WIN_RATE_WEIGHTED      = 0.48
MIN_PATTERNS               = 1     # require at least this many active patterns

KLINE_FETCH_LIMIT    = 50
PATTERN_LOG_INTERVAL = 20          # ticks between pattern status logs
MAX_LEVERAGE         = 10
RISK_PER_TRADE_PCT   = 30
TP_MULTIPLIER        = 1.8
SL_MULTIPLIER        = 0.7
MAX_HOLD_SECONDS     = 240
COOLDOWN_SECONDS     = 45
MIN_NOTIONAL         = 10
MAX_TRADES_PER_HOUR  = 5
POLL_INTERVAL_SEC    = 2

# Journal API
JOURNAL_URL  = "https://journal-cal.vercel.app"
JOURNAL_PASS = "vest2026"

# Logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Stats path
STATS_PATH = Path(__file__).parent.parent / "data" / "pattern_stats.json"


# ── Data Structures ───────────────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    side: str          # BUY/SELL
    entry_price: float
    quantity: float
    entry_time: float
    tp_price: float
    sl_price: float
    active_patterns: list
    confidence: float
    consensus: float
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


# ── Logging ───────────────────────────────────────────────────────────────

def setup_logging():
    log_file = LOG_DIR / f"pattern_scalper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ]
    )
    return logging.getLogger("pattern_scalper")


# ── Journal Client ────────────────────────────────────────────────────────

class JournalClient:
    def __init__(self, base_url=JOURNAL_URL, password=JOURNAL_PASS):
        self.base_url = base_url
        self.session = requests.Session()
        self._login(password)

    def _login(self, password):
        try:
            r = self.session.get(f"{self.base_url}/login?pass={password}", allow_redirects=False)
            if r.status_code in (301, 302):
                self.session.cookies.update(r.cookies)
            self.session.get(self.base_url)
        except Exception as e:
            logging.warning(f"Journal login failed: {e}")

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


# ── Live Data Fetching ────────────────────────────────────────────────────

def fetch_live_klines(symbol: str, interval: str, limit: int = 50) -> Optional[list]:
    """
    Fetch live klines from Orderly public API (no auth required).
    Returns list of dicts sorted oldest→newest, or None on error.
    """
    try:
        r = requests.get(
            "https://api-evm.orderly.org/v1/public/kline",
            params={"symbol": symbol, "type": interval, "limit": limit},
            timeout=10,
        )
        data = r.json()
        if data.get("success") and data.get("data", {}).get("rows"):
            rows = data["data"]["rows"]
            # API returns newest-first → reverse for oldest-first
            return list(reversed(rows))
    except Exception as e:
        logging.warning(f"kline fetch error ({interval}): {e}")
    return None


def klines_to_df(rows: list) -> "pd.DataFrame":
    """Convert Orderly kline rows to pandas DataFrame."""
    import pandas as pd
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Rename to match internal convention
    col_map = {
        "start_timestamp": "open_time",
        "end_timestamp":   "close_time",
    }
    df.rename(columns=col_map, inplace=True)
    # Ensure required columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = 0.0
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_live_trades(symbol: str, limit: int = 100) -> list:
    """Fetch recent market trades for order flow analysis."""
    try:
        r = requests.get(
            "https://api-evm.orderly.org/v1/public/market_trades",
            params={"symbol": symbol, "limit": limit},
            timeout=5,
        )
        data = r.json()
        if data.get("data", {}).get("rows"):
            return data["data"]["rows"]
    except Exception as e:
        logging.warning(f"Trades fetch error: {e}")
    return []


# ── Pattern Scalper Engine ────────────────────────────────────────────────

class PatternScalperEngine:
    def __init__(self, client: OrderlyClient, logger):
        self.client = client
        self.log = logger
        self.engine = PatternEngine(stats_path=STATS_PATH)
        self.journal: Optional[JournalClient] = None
        self.position: Optional[Position] = None
        self.stats = TradeStats()
        self.last_trade_time = 0.0
        self.running = False
        self.tick_count = 0
        self.last_composite: Optional[CompositeSignal] = None

        try:
            self.journal = JournalClient()
            self.log.info("📒 Connected to trade journal")
        except Exception:
            self.log.warning("📒 Journal unavailable")

    # ── Data fetching ──────────────────────────────────────────────────────

    def fetch_data(self):
        """Fetch all required market data. Returns (df_1m, df_5m, df_15m, trades)."""
        import pandas as pd
        rows_1m  = fetch_live_klines(SYMBOL, "1m",  KLINE_FETCH_LIMIT)
        rows_5m  = fetch_live_klines(SYMBOL, "5m",  KLINE_FETCH_LIMIT)
        rows_15m = fetch_live_klines(SYMBOL, "15m", KLINE_FETCH_LIMIT)
        trades   = fetch_live_trades(SYMBOL)

        df_1m  = klines_to_df(rows_1m)  if rows_1m  else pd.DataFrame()
        df_5m  = klines_to_df(rows_5m)  if rows_5m  else pd.DataFrame()
        df_15m = klines_to_df(rows_15m) if rows_15m else pd.DataFrame()

        return df_1m, df_5m, df_15m, trades

    def latest_price(self, df_1m) -> Optional[float]:
        if df_1m is None or len(df_1m) == 0:
            return None
        try:
            return float(df_1m.iloc[-1]["close"])
        except Exception:
            return None

    # ── Position sizing ────────────────────────────────────────────────────

    def calculate_position_size(self, price: float, confidence: float) -> float:
        """Scale size by confidence: base_risk * (0.5 + confidence * 0.5)."""
        try:
            bal = self.client.get_balance()
            holding = 0.0
            for h in bal.get("data", {}).get("holding", []):
                if h["token"] == "USDC":
                    holding = float(h["holding"])
                    break

            size_factor = 0.5 + confidence * 0.5
            risk_usd = holding * (RISK_PER_TRADE_PCT / 100) * size_factor
            notional = risk_usd * MAX_LEVERAGE

            if notional < MIN_NOTIONAL:
                self.log.warning(f"Notional ${notional:.2f} below min ${MIN_NOTIONAL}")
                return 0.0

            qty = math.floor(notional / price * 100) / 100
            return max(qty, 0.01)

        except Exception as e:
            self.log.error(f"Position size error: {e}")
            return 0.0

    # ── Trade logic ────────────────────────────────────────────────────────

    def can_trade(self) -> bool:
        if self.position:
            return False
        if time.time() - self.last_trade_time < COOLDOWN_SECONDS:
            return False
        if time.time() - self.stats.hour_start > 3600:
            self.stats.trades_this_hour = 0
            self.stats.hour_start = time.time()
        if self.stats.trades_this_hour >= MAX_TRADES_PER_HOUR:
            return False
        return True

    def should_enter(self, composite: CompositeSignal) -> bool:
        """Evaluate entry conditions from composite signal."""
        if composite.direction == "NEUTRAL":
            return False
        if composite.confidence <= ENTRY_CONFIDENCE_THRESHOLD:
            return False
        if composite.consensus < MIN_CONSENSUS:
            return False
        if composite.win_rate_weighted < MIN_WIN_RATE_WEIGHTED:
            return False
        if len(composite.active_patterns) < MIN_PATTERNS:
            return False
        return True

    def open_position(self, composite: CompositeSignal, price: float):
        """Place market order based on pattern signal."""
        direction = composite.direction
        confidence = composite.confidence
        active_patterns = composite.active_patterns
        consensus = composite.consensus

        qty = self.calculate_position_size(price, confidence)
        if qty <= 0:
            self.log.warning("⚠️  Cannot open — insufficient balance or size too small")
            return

        # ATR-based TP/SL: use avg expected move from signals
        avg_expected = sum(s.expected_move_pct for s in composite.signals) / max(len(composite.signals), 1)
        move_dollars = price * (max(avg_expected, 0.2) / 100)

        if direction == "BUY":
            tp = round(price + move_dollars * TP_MULTIPLIER, 3)
            sl = round(price - move_dollars * SL_MULTIPLIER, 3)
        else:
            tp = round(price - move_dollars * TP_MULTIPLIER, 3)
            sl = round(price + move_dollars * SL_MULTIPLIER, 3)

        self.log.info(
            f"🎯 OPENING {direction} {qty} SOL @ ~${price:.3f} | "
            f"TP: ${tp:.3f} SL: ${sl:.3f} | "
            f"Confidence: {confidence:.2f} | Consensus: {consensus:.2f} | "
            f"Patterns: {active_patterns}"
        )

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
                    tp_price=tp,
                    sl_price=sl,
                    active_patterns=list(active_patterns),
                    confidence=confidence,
                    consensus=consensus,
                    order_id=order_id,
                )
                self.last_trade_time = time.time()
                self.stats.total += 1
                self.stats.trades_this_hour += 1

                self.log.info(f"✅ Order placed! ID: {order_id}")

                # Journal entry
                if self.journal:
                    thesis = (
                        f"Patterns: {active_patterns} | "
                        f"Confidence: {confidence:.2f} | "
                        f"Consensus: {consensus:.2f}"
                    )
                    jid = self.journal.log_entry({
                        "symbol": "SOL-PERP",
                        "direction": "LONG" if direction == "BUY" else "SHORT",
                        "timeframe": "1m",
                        "strategy": "PatternScalp",
                        "entry_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "entry_time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                        "entry_price": price,
                        "stop_loss": sl,
                        "target_price": tp,
                        "position_size": qty,
                        "leverage": MAX_LEVERAGE,
                        "risk_amount": round(qty * price / MAX_LEVERAGE, 4),
                        "thesis": thesis,
                        "status": "open",
                        "tags": ["auto", "scalp", "pattern", *active_patterns],
                    })
                    if jid:
                        self.position.journal_id = str(jid)
                        self.log.info(f"📒 Journaled as trade #{jid}")
            else:
                self.log.error(f"❌ Order failed: {result}")

        except Exception as e:
            self.log.error(f"❌ Order error: {e}")

    def check_exit(self, current_price: float):
        """Check TP/SL/timeout exit conditions."""
        if not self.position:
            return

        pos = self.position
        elapsed = time.time() - pos.entry_time
        exit_reason = None
        exit_type = None

        if pos.side == "BUY" and current_price >= pos.tp_price:
            exit_reason = f"Take profit hit (${pos.tp_price:.3f})"
            exit_type = "tp"
        elif pos.side == "SELL" and current_price <= pos.tp_price:
            exit_reason = f"Take profit hit (${pos.tp_price:.3f})"
            exit_type = "tp"
        elif pos.side == "BUY" and current_price <= pos.sl_price:
            exit_reason = f"Stop loss hit (${pos.sl_price:.3f})"
            exit_type = "sl"
        elif pos.side == "SELL" and current_price >= pos.sl_price:
            exit_reason = f"Stop loss hit (${pos.sl_price:.3f})"
            exit_type = "sl"
        elif elapsed >= MAX_HOLD_SECONDS:
            exit_reason = f"Max hold time ({MAX_HOLD_SECONDS}s)"
            exit_type = "timeout"

        if exit_reason:
            self.close_position(current_price, exit_reason, exit_type)

    def close_position(self, current_price: float, reason: str, exit_type: str):
        """Close position, update stats, update pattern engine."""
        if not self.position:
            return

        pos = self.position
        close_side = "SELL" if pos.side == "BUY" else "BUY"

        self.log.info(
            f"🔄 CLOSING {close_side} {pos.quantity} SOL @ ~${current_price:.3f} | {reason}"
        )

        try:
            result = self.client.create_order(
                symbol=SYMBOL,
                side=close_side,
                order_type="MARKET",
                order_quantity=pos.quantity,
                reduce_only=True,
            )

            if result.get("success"):
                if pos.side == "BUY":
                    pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                else:
                    pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100

                pnl_usd = pos.quantity * abs(current_price - pos.entry_price)
                if pnl_pct < 0:
                    pnl_usd = -pnl_usd

                won = pnl_usd > 0
                if won:
                    self.stats.wins += 1
                else:
                    self.stats.losses += 1
                self.stats.total_pnl += pnl_usd

                emoji = "🟢" if won else "🔴"
                self.log.info(
                    f"{emoji} P&L: ${pnl_usd:+.4f} ({pnl_pct:+.3f}%) | Reason: {reason}"
                )
                self.log.info(
                    f"   Active patterns: {pos.active_patterns} | "
                    f"Confidence: {pos.confidence:.2f} | Won: {won}"
                )
                self.log.info(
                    f"📊 Stats: {self.stats.wins}W/{self.stats.losses}L | "
                    f"Total P&L: ${self.stats.total_pnl:+.4f}"
                )

                # Update pattern engine with real trade result
                for pattern_name in pos.active_patterns:
                    self.engine.update_stats(
                        pattern_name=pattern_name,
                        won=won,
                        move_pct=abs(pnl_pct),
                    )
                self.log.info(
                    f"   Pattern stats updated for: {pos.active_patterns}"
                )

                # Update journal
                if self.journal and pos.journal_id:
                    error_cat = None
                    if exit_type == "sl":
                        error_cat = "bad_entry"
                    elif exit_type == "timeout":
                        error_cat = "no_follow_through"

                    hold_time = time.time() - pos.entry_time
                    self.journal.update_trade(pos.journal_id, {
                        "status": "closed",
                        "actual_entry": pos.entry_price,
                        "actual_exit": current_price,
                        "actual_pnl_pct": round(pnl_pct, 3),
                        "actual_pnl_usd": round(pnl_usd, 4),
                        "exit_reason": reason,
                        "exit_type": exit_type,
                        "error_category": error_cat,
                        "lessons": (
                            f"Patterns: {pos.active_patterns} | "
                            f"Conf: {pos.confidence:.2f} | "
                            f"Hold: {hold_time:.0f}s | Won: {won}"
                        ),
                    })

                self.position = None

            else:
                self.log.error(f"❌ Close failed: {result}")

        except Exception as e:
            self.log.error(f"❌ Close error: {e}")

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self):
        """Main polling loop."""
        self.running = True
        self.log.info(f"🚀 Pattern Scalper starting on {SYMBOL}")
        self.log.info(f"   Entry confidence: >{ENTRY_CONFIDENCE_THRESHOLD}")
        self.log.info(f"   Min consensus: {MIN_CONSENSUS}")
        self.log.info(f"   TP: {TP_MULTIPLIER}x | SL: {SL_MULTIPLIER}x")
        self.log.info(f"   Max hold: {MAX_HOLD_SECONDS}s | Cooldown: {COOLDOWN_SECONDS}s")
        self.log.info(f"   Leverage: {MAX_LEVERAGE}x | Risk/trade: {RISK_PER_TRADE_PCT}%")
        self.log.info(f"   Stats file: {STATS_PATH}")

        # Print loaded pattern stats
        self.log.info("   Loaded pattern stats:")
        for name, s in self.engine.stats.items():
            self.log.info(
                f"     {name:<28} WR={s['win_rate']:.2f} n={s['sample_count']}"
            )

        # Show balance
        try:
            bal = self.client.get_balance()
            for h in bal.get("data", {}).get("holding", []):
                if h["token"] == "USDC":
                    self.log.info(f"   💰 Balance: ${h['holding']:.4f} USDC")
        except Exception:
            pass

        while self.running:
            try:
                self.tick_count += 1

                # Fetch live data
                df_1m, df_5m, df_15m, trades = self.fetch_data()
                current_price = self.latest_price(df_1m)

                if current_price is None:
                    self.log.warning("Could not determine current price, skipping tick")
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                # Run pattern engine
                composite = self.engine.analyze(df_1m, df_5m, df_15m, trades)
                self.last_composite = composite

                # Periodic pattern status log
                if self.tick_count % PATTERN_LOG_INTERVAL == 0:
                    self.log.info(
                        f"📈 SOL ${current_price:.3f} | "
                        f"Pattern: {composite.direction} "
                        f"conf={composite.confidence:.2f} "
                        f"consensus={composite.consensus:.2f} | "
                        f"Active: {composite.active_patterns or 'none'}"
                    )
                    for sig in composite.signals:
                        self.log.info(
                            f"   {sig.name}: {sig.direction} "
                            f"conf={sig.raw_confidence:.2f} "
                            f"features={sig.features}"
                        )

                # Manage open position
                if self.position:
                    self.check_exit(current_price)

                # Check for new entry
                elif self.can_trade() and self.should_enter(composite):
                    self.log.info(
                        f"⚡ Pattern signal: {composite.direction} | "
                        f"conf={composite.confidence:.2f} | "
                        f"consensus={composite.consensus:.2f} | "
                        f"win_rate_wt={composite.win_rate_weighted:.2f} | "
                        f"patterns={composite.active_patterns}"
                    )
                    self.open_position(composite, current_price)

                time.sleep(POLL_INTERVAL_SEC)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log.error(f"Loop error: {e}", exc_info=True)
                time.sleep(5)

        # Clean up open position on shutdown
        if self.position:
            self.log.info("Shutdown: closing open position...")
            df_1m, _, _, _ = self.fetch_data()
            price = self.latest_price(df_1m)
            if price:
                self.close_position(price, "Shutdown", "shutdown")

        self.log.info(
            f"🛑 Scalper stopped. "
            f"Final: {self.stats.wins}W/{self.stats.losses}L | "
            f"P&L: ${self.stats.total_pnl:+.4f}"
        )


# ── Entry Point ───────────────────────────────────────────────────────────

def main():
    logger = setup_logging()

    logger.info("Connecting to Orderly...")
    client = OrderlyClient(SOLANA_KEY)
    if not client.is_ready:
        logger.info("Running first-time setup...")
        client.setup()

    logger.info(
        f"✅ Connected as {client.address[:12]}... | Account: {client.account_id[:16]}..."
    )

    # Smoke-test: micro trade
    logger.info("🧪 Running micro trade test...")
    try:
        client.test_micro_trade(SYMBOL, MIN_NOTIONAL)
        logger.info("✅ Micro trade test passed")
    except Exception as e:
        logger.error(f"❌ Micro trade test failed: {e}")
        return

    scalper = PatternScalperEngine(client, logger)

    def handle_signal(sig, frame):
        logger.info("Received shutdown signal...")
        scalper.running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    scalper.run()


if __name__ == "__main__":
    main()
