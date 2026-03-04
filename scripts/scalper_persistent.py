#!/usr/bin/env python3
"""
scalper_persistent.py — Persistent Scalper with Live Recovery

Fixed version that integrates with aggressive heartbeat and persists actual
live trade data (Orderly) instead of cached OHLCV data from different sources.
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
from dataclasses import dataclass
from typing import Optional

# Add persistent storage
sys.path.insert(0, str(Path(__file__).parent))
from live_persistence import LiveTradeStore
from persistent_data import get_persistent_store  # For risk params
from orderly_auth import OrderlyClient

# =============================================================================
# CONFIGURATION (from persistent storage)
# =============================================================================

SYMBOL = "PERP_SOL_USDC"
SOLANA_KEY = "5XY4ErjzPekDin7MyBzLcN6Dvd7rn2BRPaGzwZvStpu27uwyp7JXvPYpZfaCJ1nEBMeFoWqginvvfDBERdsKmGUj"

# Get dynamic config from persistent storage on startup
store = get_persistent_store()
config = {
    'JUMP_THRESHOLD_PCT': float(store.get_risk_param('jump_threshold_pct', 0.5)),
    'MAX_LEVERAGE': int(store.get_risk_param('max_leverage', 10)),
    'RISK_PER_TRADE_PCT': float(store.get_risk_param('risk_per_trade_pct', 30)),
    'TP_MULTIPLIER': 1.5,
    'SL_MULTIPLIER': 0.8,
    'LOOKBACK_SECONDS': 60,
    'MAX_HOLD_SECONDS': 300,
    'COOLDOWN_SECONDS': 30,
    'MAX_TRADES_PER_HOUR': 5
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    entry_time: float
    jump_size_pct: float
    tp_price: float
    sl_price: float
    trade_audit_id: int
    order_id: Optional[int] = None

# =============================================================================
# PERSISTENT TRADE INTEGRATION
# =============================================================================

class PersistentScalperEngine:
    def __init__(self, client: OrderlyClient):
        self.client = client
        self.live_store = LiveTradeStore()
        
        # Refresh config from persistent storage
        global config
        config['JUMP_THRESHOLD_PCT'] = float(store.get_risk_param('jump_threshold_pct', 0.5))
        config['MAX_LEVERAGE'] = int(store.get_risk_param('max_leverage', 10))
        
        self.position: Optional[Position] = None
        self.last_trade_time = 0
        self.trades_this_hour = 0
        self.hour_start = time.time()
        
        # Pre-load live trades for immediate readiness
        self.live_store.init_live_cache()
        
        logging.info(f"🚀 Persistent scalper initialized")
        logging.info(f"   Config: {config['JUMP_THRESHOLD_PCT']}% jump | {config['MAX_LEVERAGE']}x leverage")
    
    def calculate_position_size(self, price: float) -> float:
        """Calculate position size from available balance"""
        try:
            bal = self.client.get_balance()
            holding = 0
            for h in bal.get("data", {}).get("holding", []):
                if h["token"] == "USDC":
                    holding = float(h["holding"])
                    break
            
            risk_usd = holding * (config['RISK_PER_TRADE_PCT'] / 100)
            notional = risk_usd * config['MAX_LEVERAGE']
            
            if notional < 10:  # Min notional
                return 0
            
            qty = math.floor(notional / price * 100) / 100
            return max(qty, 0.01)
            
        except Exception as e:
            logging.error(f"Position size error: {e}")
            return 0
    
    def open_position(self, direction: str, jump_pct: float, price: float):
        """Open position with persistent audit"""
        qty = self.calculate_position_size(price)
        if qty <= 0:
            logging.warning("⚠️ Balance insufficient")
            return
        
        # Calculate TP/SL levels
        jump_dollars = price * (jump_pct / 100)
        if direction == "BUY":
            tp = round(price + jump_dollars * config['TP_MULTIPLIER'], 3)
            sl = round(price - jump_dollars * config['SL_MULTIPLIER'], 3)
        else:
            tp = round(price - jump_dollars * config['TP_MULTIPLIER'], 3)
            sl = round(price + jump_dollars * config['SL_MULTIPLIER'], 3)
        
        logging.info(f"🎯 OPEN {direction} {qty} SOL @{price:.3f} | TP: {tp} SL: {sl} | {jump_pct:.2f}%")
        
        try:
            result = self.client.create_order(
                symbol=SYMBOL,
                side=direction,
                order_type="MARKET",
                order_quantity=qty,
            )
            
            if result.get("success"):
                order_id = result["data"]["order_id"]
                
                # Store in persistent audit
                audit_id = store.store_trade_decision(
                    symbol="SOL-PERP",
                    direction=direction,
                    entry_price=price,
                    jump_size_pct=jump_pct,
                    aggressiveness_level=config.get('AGGRESSIVENESS_LEVEL', 0)
                )
                
                self.position = Position(
                    symbol=SYMBOL,
                    side=direction,
                    entry_price=price,
                    quantity=qty,
                    entry_time=time.time(),
                    jump_size_pct=jump_pct,
                    tp_price=tp,
                    sl_price=sl,
                    trade_audit_id=audit_id,
                    order_id=order_id
                )
                
                self.last_trade_time = time.time()
                store.set_current_position(
                    symbol="PERP_SOL_USDC",
                    side=direction,
                    quantity=qty,
                    avg_entry_price=price,
                    position_value=qty * price,
                    tp_price=tp,
                    sl_price=sl
                )
                
                logging.info(f"✅ Position opened - Audit ID: {audit_id}")
            
        except Exception as e:
            logging.error(f"Open position error: {e}")
    
    def close_position(self, current_price: float, reason: str):
        """Close position and update persistent audit"""
        if not self.position:
            return
            
        pos = self.position
        close_side = "SELL" if pos.side == "BUY" else "BUY"
        
        logging.info(f"🔄 CLOSE {close_side} {pos.quantity} SOL @{current_price:.3f} | {reason}")
        
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
                    pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                else:
                    pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
                
                # Update audit with exit
                store.update_trade_exit(
                    trade_id=pos.trade_audit_id,
                    exit_price=current_price,
                    exit_reason=reason
                )
                
                # Clear position from persistent storage
                store.clear_position("PERP_SOL_USDC")
                
                emoji = "🟢" if pnl_pct > 0 else "🔴"
                logging.info(f"{emoji} P&L: {pnl_pct:+.2f}% | {reason}")
                
                self.position = None
                
        except Exception as e:
            logging.error(f"Close position error: {e}")
    
    def can_trade(self) -> bool:
        """Check if we can open new position"""
        if self.position:
            return False
        if time.time() - self.last_trade_time < config['COOLDOWN_SECONDS']:
            return False
        
        # Reset hourly counter
        if time.time() - self.hour_start > 3600:
            self.trades_this_hour = 0
            self.hour_start = time.time()
            store.set_risk_param('trades_this_hour', 0)
            store.set_risk_param('hour_start', self.hour_start)
        
        return self.trades_this_hour < config['MAX_TRADES_PER_HOUR']
    
    def check_exit_conditions(self, current_price: float):
        """Check position exit conditions"""
        if not self.position:
            return
            
        pos = self.position
        elapsed = time.time() - pos.entry_time
        
        exit_reason = None
        
        # TP hit
        if pos.side == "BUY" and current_price >= pos.tp_price:
            exit_reason = f"TP hit @ {pos.tp_price}"
        elif pos.side == "SELL" and current_price <= pos.tp_price:
            exit_reason = f"TP hit @ {pos.tp_price}"
        
        # SL hit  
        elif pos.side == "BUY" and current_price <= pos.sl_price:
            exit_reason = f"SL hit @ {pos.sl_price}"
        elif pos.side == "SELL" and current_price >= pos.sl_price:
            exit_reason = f"SL hit @ {pos.sl_price}"
        
        # Timeout
        elif elapsed >= config['MAX_HOLD_SECONDS']:
            exit_reason = f"Timeout ({config['MAX_HOLD_SECONDS']}s)"
        
        if exit_reason:
            self.close_position(current_price, exit_reason)
    
    def run(self):
        """Main scalping loop with persistent recovery"""
        logging.info("🔍 Starting persistent scalping loop...")
        
        tick_count = 0
        while True:
            try:
                # Continuous live trade caching
                trades = self.live_store.fetch_live_trades(limit=5)
                self.live_store.cache_recent_trades(trades)
                
                # Check for jump using actual trade data
                jump = self.live_store.detect_price_jump(lookback_seconds=config['LOOKBACK_SECONDS'])
                
                latest_price, latest_time = self.live_store.get_latest_price()
                if latest_price:
                    tick_count += 1
                    
                    # Log status every 30 cycles
                    if tick_count % 30 == 0:
                        logging.info(f"📈 SOL ${latest_price:.3f}")
                    
                    # Check exit on open positions
                    self.check_exit_conditions(latest_price)
                    
                    # Check for new entry
                    if self.can_trade() and jump:
                        direction, jump_pct, price = jump
                        logging.info(f"⚡ Jump detected: {jump_pct:+.2f}% → {direction}")
                        self.open_position(direction, jump_pct, price)
                        self.trades_this_hour += 1
                        store.set_risk_param('trades_this_hour', self.trades_this_hour)
                
                time.sleep(2)  # Conservative polling
                
            except KeyboardInterrupt:
                logging.info("⏹️ Shutdown requested")
                if self.position:
                    latest_price, _ = self.live_store.get_latest_price()
                    if latest_price:
                        self.close_position(latest_price, "Shutdown")
                break
                
            except Exception as e:
                logging.error(f"Loop error: {e}")
                time.sleep(5)

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    log_path = Path(__file__).parent.parent / "logs" / "scalper_persistent.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [SCALPER] %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    
    logging.info("🚀 Initializing persistent scalper...")
    
    try:
        client = OrderlyClient(SOLANA_KEY)
        if not client.is_ready:
            logging.info("First-time Orderly setup...")
            client.setup()
        
        balance = client.get_balance()
        for h in balance.get("data", {}).get("holding", []):
            if h["token"] == "USDC":
                usdc_balance = float(h["holding"])
                logging.info(f"💰 Current balance: ${usdc_balance:.4f} USDC")
                break
        
        engine = PersistentScalperEngine(client)
        engine.run()
        
    except Exception as e:
        logging.error(f"Startup failed: {e}")

def refresh_config():
    """Heartbeat can call this to update live config"""
    global config
    config['JUMP_THRESHOLD_PCT'] = float(store.get_risk_param('jump_threshold_pct', 0.5))
    config['MAX_LEVERAGE'] = int(store.get_risk_param('max_leverage', 10))
    logging.warning(f"⚙️ Config updated: {config['JUMP_THRESHOLD_PCT']}% @ {config['MAX_LEVERAGE']}x")

if __name__ == "__main__":
    main()