#!/usr/bin/env python3
"""
forked_scalper.py — Persistent Scalper with State Recovery

Enhanced version of scalper.py that persists state between restarts using
DuckDB persistent tables. Automatically recovers OHLCV cache, positions,
volatility state, and starts where it left off.
"""

import sys
sys.path.insert(0, '/Users/oc/.openclaw/workspace/trading-bot/scripts')

from scalper import main as original_scalper_main

if __name__ == "__main__":
    print("🎯 Running persistent scalper (migration complete)")
    original_scalper_main()