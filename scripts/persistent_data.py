#!/usr/bin/env python3
"""
persistent_data.py — Persist sliding window data and restart state

Stores in-memory data to DuckDB persistent tables so scalper can resume
state after restarts. Handles OHLCV caching, trade history, and risk tracking.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
import duckdb
import pandas as pd
from typing import Dict, List, Optional, Any

WORKSPACE = Path("/Users/oc/.openclaw/workspace/apps/trading-bot")
PERSISTENT_DB = WORKSPACE / "data" / "trading.duckdb"
CONFIG_DIR = WORKSPACE / "config"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("persistent_data")

class PersistentDataStore:
    """Manages persistent storage for scalper restart state"""
    
    def __init__(self):
        self.db_path = PERSISTENT_DB
        self.config_dir = CONFIG_DIR
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize DuckDB with persistent connection
        self.con = duckdb.connect(str(self.db_path))
        self._init_schema()
    
    def _init_schema(self):
        """Create persistent data tables"""
        
        # Main OHLCV cache - rolling 48h window
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS kline_cache (
                symbol VARCHAR NOT NULL,
                open_time BIGINT NOT NULL,      -- unix timestamp (ms)
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE DEFAULT 0,
                close_time BIGINT,
                PRIMARY KEY (symbol, open_time)
            )
        """)
        
        # Recent trade decisions and outcomes
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS trade_audit (
                id BIGINT DEFAULT (epoch_ms(now())),
                symbol VARCHAR,
                direction VARCHAR,
                entry_price DOUBLE,
                entry_time BIGINT,
                jump_size_pct DOUBLE,
                exit_price DOUBLE,
                exit_time BIGINT,
                pnl_pct DOUBLE,
                pnl_usd DOUBLE,
                holding_duration_s INT,
                exit_reason VARCHAR,
                aggressiveness_level INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Volatility regime state (volatility buckets)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS volatility_regime (
                symbol VARCHAR PRIMARY KEY,
                current_regime VARCHAR,         -- low / medium / high
                regime_change_time BIGINT,
                realized_vol_1h DOUBLE,
                realized_vol_4h DOUBLE,
                realized_vol_24h DOUBLE,
                last_updated BIGINT
            )
        """)
        
        # Position tracking (current exposure)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS open_positions (
                symbol VARCHAR PRIMARY KEY,
                side VARCHAR,                   -- BUY / SELL / LONG / SHORT
                quantity DECIMAL,
                avg_entry_price DECIMAL,
                position_value DECIMAL,
                unrealized_pnl DECIMAL,
                opened_at BIGINT,
                last_update BIGINT,
                tp_price DECIMAL,
                sl_price DECIMAL
            )
        """)
        
        # Risk parameters state
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS risk_params (
                key VARCHAR PRIMARY KEY,
                value VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        logger.info("✅ Persistent data schema initialized")
    
    def store_kline(self, symbol: str, open_time: int, open_price: float, 
                   high: float, low: float, close: float, volume: float = 0,
                   close_time: int = None):
        """Store single kline data point with deduplication"""
        self.con.execute("""
            INSERT OR REPLACE INTO kline_cache 
            (symbol, open_time, open, high, low, close, volume, close_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, open_time, open_price, high, low, close, volume, close_time or open_time))
    
    def store_trade_decision(self, symbol: str, direction: str, entry_price: float,
                            jump_size_pct: float, aggressiveness_level: int = 0) -> int:
        """Store new trade decision (before exit)"""
        entry_time = int(time.time() * 1000)  # ms
        
        self.con.execute("""
            INSERT INTO trade_audit 
            (symbol, direction, entry_price, entry_time, jump_size_pct, aggressiveness_level)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING id
        """, (symbol, direction, entry_price, entry_time, jump_size_pct, aggressiveness_level))
        
        result = self.con.fetchone()
        return result[0] if result else None
    
    def update_trade_exit(self, trade_id: int, exit_price: float, exit_reason: str):
        """Update trade with exit details"""
        exit_time = int(time.time() * 1000)  # ms
        
        # Get entry details
        self.con.execute("""
            SELECT entry_price, direction, entry_time FROM trade_audit WHERE id = ?
        """, (trade_id,))
        
        result = self.con.fetchone()
        if not result:
            logger.warning(f"Trade ID {trade_id} not found")
            return
            
        entry_price, direction, entry_time = result
        
        # Calculate P&L
        if direction.upper() in ['BUY', 'LONG']:
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            
        # For now, we'll approximate USD P&L based on 1 SOL (real trading will have actual amounts)
        pnl_usd = abs(pnl_pct) * 20  # Approx $20 per 1% move per SOL
        
        holding_duration = (exit_time - entry_time) // 1000  # seconds
        
        self.con.execute("""
            UPDATE trade_audit SET
                exit_price = ?,
                exit_time = ?,
                pnl_pct = ?,
                pnl_usd = ?,
                holding_duration_s = ?,
                exit_reason = ?,
                created_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (exit_price, exit_time, pnl_pct, pnl_usd, holding_duration, exit_reason, trade_id))
    
    def cache_recent_prices(self, symbol: str, prices: List[Dict[str, Any]]):
        """Cache recent price ticks for restart recovery"""
        for price_data in prices:
            self.store_kline(
                symbol=symbol,
                open_time=price_data['timestamp'],
                open_price=price_data['price'],
                high=price_data.get('high', price_data['price']),
                low=price_data.get('low', price_data['price']),
                close=price_data['price'],
                volume=price_data.get('volume', 0)
            )
    
    def set_volatility_regime(self, symbol: str, regime: str, vol_1h: float, 
                            vol_4h: float, vol_24h: float):
        """Store current volatility regime"""
        current_time = int(time.time())
        
        self.con.execute("""
            INSERT OR REPLACE INTO volatility_regime 
            (symbol, current_regime, regime_change_time, 
             realized_vol_1h, realized_vol_4h, realized_vol_24h, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, regime, current_time, vol_1h, vol_4h, vol_24h, current_time))
    
    def set_current_position(self, symbol: str, side: str, quantity: float,
                           avg_entry_price: float, position_value: float,
                           unrealized_pnl: float = 0, tp_price: float = None,
                           sl_price: float = None):
        """Store current open position"""
        current_time = int(time.time())
        
        self.con.execute("""
            INSERT OR REPLACE INTO open_positions 
            (symbol, side, quantity, avg_entry_price, position_value, 
             unrealized_pnl, opened_at, last_update, tp_price, sl_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, side, quantity, avg_entry_price, position_value, 
               unrealized_pnl, current_time, current_time, tp_price, sl_price))
    
    def clear_position(self, symbol: str):
        """Remove closed position from storage"""
        self.con.execute("DELETE FROM open_positions WHERE symbol = ?", (symbol,))
    
    def set_risk_param(self, key: str, value: Any):
        """Store risk parameters like jump_threshold_pct, max_leverage etc"""
        str_value = str(value)
        self.con.execute("""
            INSERT OR REPLACE INTO risk_params (key, value)
            VALUES (?, ?)
        """, (key, str_value))
    
    def get_risk_param(self, key: str, default: Any = None) -> Any:
        """Retrieve risk parameter"""
        self.con.execute("SELECT value FROM risk_params WHERE key = ?", (key,))
        result = self.con.fetchone()
        return result[0] if result else default
    
    def load_recent_ohlcv(self, symbol: str, lookback_hours: int = 2) -> pd.DataFrame:
        """Load recent OHLCV data for restart window"""
        cutoff_time = int((datetime.now() - timedelta(hours=lookback_hours)).timestamp() * 1000)
        
        return self.con.execute("""
            SELECT * FROM kline_cache 
            WHERE symbol = ? AND open_time >= ?
            ORDER BY open_time ASC
        """, (symbol, cutoff_time)).df()
    
    def load_recent_trades(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """Load recent trade history"""
        return self.con.execute("""
            SELECT * FROM trade_audit 
            WHERE symbol = ?
            ORDER BY entry_time DESC
            LIMIT ?
        """, (symbol, limit)).df()
    
    def get_volatility_regime(self, symbol: str) -> Dict[str, Any]:
        """Load current volatility regime"""
        result = self.con.execute("""
            SELECT current_regime, realized_vol_1h, realized_vol_4h, 
                   realized_vol_24h, last_updated
            FROM volatility_regime WHERE symbol = ?
        """, (symbol,)).fetchone()
        
        if result:
            return {
                'regime': result[0],
                'vol_1h': result[1],
                'vol_4h': result[2],
                'vol_24h': result[3],
                'last_updated': result[4]
            }
        return {'regime': 'medium', 'vol_1h': 0, 'vol_4h': 0, 'vol_24h': 0, 'last_updated': 0}
    
    def get_current_position(self, symbol: str) -> Dict[str, Any]:
        """Load current open position"""
        result = self.con.execute("""
            SELECT * FROM open_positions WHERE symbol = ?
        """, (symbol,)).fetchone()
        
        if result:
            return {
                'side': result[1],
                'quantity': result[2],
                'avg_entry_price': result[3],
                'position_value': result[4],
                'unrealized_pnl': result[5],
                'opened_at': result[6],
                'tp_price': result[8],
                'sl_price': result[9]
            }
        return None
    
    def cleanup_old_data(self, retention_hours: int = 48):
        """Clean up old cached data"""
        cutoff_time = int((datetime.now() - timedelta(hours=retention_hours)).timestamp() * 1000)
        
        # Clean old OHLCV cache
        self.con.execute("DELETE FROM kline_cache WHERE open_time < ?", (cutoff_time,))
        
        # Clean very old trade audit (retain last 100 trades)
        self.con.execute("""
            DELETE FROM trade_audit 
            WHERE id NOT IN (
                SELECT id FROM trade_audit 
                ORDER BY id DESC 
                LIMIT 100
            )
        """)
        
        logger.info(f"🧹 Cleaned old data - Retention: {retention_hours}h")
    
    def get_restart_context(self, symbol: str) -> Dict[str, Any]:
        """Get complete context for scalper restart"""
        return {
            'recent_ohlcv': self.load_recent_ohlcv(symbol).to_dict('records'),
            'current_position': self.get_current_position(symbol),
            'volatility_regime': self.get_volatility_regime(symbol),
            'recent_trades': self.load_recent_trades(symbol).to_dict('records'),
            'risk_params': {
                'jump_threshold_pct': float(self.get_risk_param('jump_threshold_pct', 0.5)),
                'max_leverage': int(self.get_risk_param('max_leverage', 10)),
                'risk_per_trade_pct': float(self.get_risk_param('risk_per_trade_pct', 30))
            },
            'loaded_at': int(time.time())
        }
    
    def dump_for_debug(self, symbol: str):
        """Dump all persistent data for debugging"""
        context = self.get_restart_context(symbol)
        debug_path = self.config_dir / f"scalper_debug_{int(time.time())}.json"
        debug_path.write_text(json.dumps(context, indent=2))
        logger.info(f"💾 Debug data dumped to {debug_path}")
    
    def close(self):
        """Close database connection"""
        self.con.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

# Global singleton for easy access
_store = None

def get_persistent_store() -> PersistentDataStore:
    """Get or create the singleton persistent datastore"""
    global _store
    if _store is None:
        _store = PersistentDataStore()
    return _store