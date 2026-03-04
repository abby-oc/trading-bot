"""
signals.py — Print current live signals from all strategies.

Usage:
    python3 scripts/signals.py
    python3 scripts/signals.py --symbol BTC-PERP
    python3 scripts/signals.py --json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

from market_db import MarketDB
from strategy  import STRATEGIES, RatioZScore


def get_all_signals(db: MarketDB) -> list[dict]:
    signals = []

    for name, strat in STRATEGIES.items():
        if isinstance(strat, RatioZScore):
            df  = db.ohlcv("BTCSOL", strat.default_interval, limit=500)
            sym = "BTCSOL"
        else:
            for sym in ["BTC-PERP", "SOL-PERP"]:
                df = db.ohlcv(sym, strat.default_interval, limit=500)
                if df.empty:
                    continue
                sig = strat.current_signal(df, sym)
                signals.append({
                    "strategy":  sig.strategy,
                    "symbol":    sig.symbol,
                    "direction": sig.direction,
                    "strength":  round(sig.strength, 3),
                    "price":     sig.price,
                    "reason":    sig.reason,
                    "timestamp": sig.timestamp.isoformat(),
                })
            continue

        if df.empty:
            continue
        sig = strat.current_signal(df, sym)
        signals.append({
            "strategy":  sig.strategy,
            "symbol":    sig.symbol,
            "direction": sig.direction,
            "strength":  round(sig.strength, 3),
            "price":     sig.price,
            "reason":    sig.reason,
            "timestamp": sig.timestamp.isoformat(),
        })

    return signals


def print_signals(signals: list[dict]):
    icon = {"LONG": "▲ ", "SHORT": "▼ ", "FLAT": "● "}
    col  = {"LONG": "\033[92m", "SHORT": "\033[91m", "FLAT": "\033[90m"}
    rst  = "\033[0m"

    print(f"\n{'═'*60}")
    print(f"  Live Signals  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'═'*60}")

    by_sym = {}
    for s in signals:
        by_sym.setdefault(s["symbol"], []).append(s)

    for sym, sigs in sorted(by_sym.items()):
        print(f"\n  {sym}")
        print(f"  {'─'*56}")
        for s in sigs:
            d   = s["direction"]
            bar = "█" * int(s["strength"] * 10) + "░" * (10 - int(s["strength"] * 10))
            print(f"  {col[d]}{icon[d]}{s['strategy']:<16}{rst}"
                  f"  {col[d]}{d:<5}{rst}"
                  f"  [{bar}] {s['strength']:.2f}"
                  f"  @ {s['price']:,.2f}")
            print(f"    {s['reason']}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    db = MarketDB(read_only=True)
    signals = get_all_signals(db)
    db.close()

    if args.json:
        print(json.dumps(signals, indent=2))
    else:
        print_signals(signals)


if __name__ == "__main__":
    main()
