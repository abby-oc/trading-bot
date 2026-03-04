#!/usr/bin/env python3
"""
live_persistence.py — Real Trade Polling Persistence

Fixes the data source mismatch by storing actual live trades from Orderly
instead of faking OHLCV candles from Vest data.
"""

import json
import time
import duckdb
import requests
from datetime import datetime, timedelta
from pathlib import Path
import logging

WORKSPACE = Path("/Users/oc/.openclaw/workspace/trading-bot")
db_path = WORKSPACE / "data" / "trading.duckdb"

class LiveTradeStore:
    """Stores actual live trades from Orderly API for restart recovery"""
    
    def __init__(self):
        self.con = duckdb.connect(str(db_path))
        self.base_url = "https://server-prod.hz.vestmarkets.com/v2"  # Orderly
        self.symbol = "PERP_SOL_USDC"
        
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS live_trade_cache (
                trade_id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                executed_price DOUBLE,
                executed_quantity DOUBLE,
                side VARCHAR,  -- BUY/SELL
                executed_timestamp BIGINT,
                ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Keep last 6h of trades for adequate lookback
        self.cleanup_old_trades()
    
    def fetch_live_trades(self, limit=50):
        """Get latest actual trades from Orderly"""
        try:
            r = requests.get(
                f"{self.base_url}/v1/public/market_trades",
                params={"symbol": self.symbol, "limit": limit},
                timeout=5
            )
            r.raise_for_status()
            data = r.json()
            
            if data.get("success"):
                return data["data"]["rows"]
        except Exception as e:
            logging.error(f"Failed to fetch live trades: {e}")
        return []
    
    def cache_recent_trades(self, trades):
        """Store live trades in persistent storage"""
        if not trades:
            return
            
        trades_df = [
            {
                'trade_id': str(t['trade_id']),
                'symbol': self.symbol,
                'executed_price': float(t['executed_price']),
                'executed_quantity': float(t['executed_quantity']),
                'side': t['side'],
                'executed_timestamp': int(t['executed_timestamp'])
            }
            for t in trades
        ]
        
        # Upsert without duplication via trade_id PK
        self.con.execute("""
            INSERT OR REPLACE INTO live_trade_cache 
            SELECT * FROM trades_df
        """, [trades_df])
    
    def get_recent_prices(self, lookback_hours=2, limit=100):
        """Get cached live trades for restart window"""
        cutoff = int((datetime.now() - timedelta(hours=lookback_hours)).timestamp() * 1000)
        
        return self.con.execute("""
            SELECT *
            FROM live_trade_cache 
            WHERE executed_timestamp >= ?
            ORDER BY executed_timestamp DESC
            LIMIT ?
        """, [cutoff, limit]).df()
    
    def detect_price_jump(self, lookback_seconds=60) -> tuple:
        """Calculate actual price change from cached trades"""
        recent_df = self.get_recent_prices(lookback_hours=1, limit=100)
        if recent_df.empty:
            return None
        
        # Convert ms to seconds
        cutoff = int(time.time() * 1000) - (lookback_seconds * 1000)
        recent_df = recent_df[recent_df['executed_timestamp'] > cutoff]
        
        if len(recent_df) < 2:
            return None
            
        # Weighted by trade size
        earliest = recent_df.iloc[-1]  # Oldest
        latest = recent_df.iloc[0]     # Newest
        
        price_change = latest['executed_price'] - earliest['executed_price']
        pct_change = (price_change / earliest['executed_price']) * 100
        
        # Ensure minimum jump
        if abs(pct_change) >= 0.5:  # 0.5% threshold
            direction = "BUY" if pct_change > 0 else "SELL"
            return (direction, abs(pct_change), latest['executed_price'])
        
        return None
    
    def get_latest_price(self):
        """Get most recent cached trade price"""
        result = self.con.execute("""
            SELECT executed_price, executed_timestamp 
            FROM live_trade_cache 
            ORDER BY executed_timestamp DESC 
            LIMIT 1
        """).fetchone()
        
        if result:
            return float(result[0]), int(result[1])
        return None, None
    
    def cleanup_old_trades(self, retention_hours=6):
        """Keep only recent trades"""
        cutoff = int((datetime.now() - timedelta(hours=retention_hours)).timestamp() * 1000)
        self.con.execute("DELETE FROM live_trade_cache WHERE executed_timestamp < ?", [cutoff])
    
    def init_live_cache(self):
        """Startup routine - pull current trades"""
        trades = self.fetch_live_trades(limit=100)
        self.cache_recent_trades(trades)
        count = len(trades)
        logging.info(f"📊 Pre-loaded {count} live trades for restart continuity")

if __name__ == "__main__":
    store = LiveTradeStore()
    print(store.get_recent_prices().to_string())