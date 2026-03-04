#!/usr/bin/env python3
"""
ingest.py — Paginated historical ingestion from Vest Markets into DuckDB.

Fetches BTC-PERP and SOL-PERP for configured intervals.
BTCSOL is a derived view — no separate ingestion needed.

Usage:
    python3 scripts/ingest.py                        # full history, all intervals
    python3 scripts/ingest.py --intervals 1d,4h      # specific intervals
    python3 scripts/ingest.py --incremental          # only fetch new candles
    python3 scripts/ingest.py --symbols BTC-PERP     # one symbol
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request, parse, error

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))
from market_db import MarketDB

BASE_URL  = "https://server-prod.hz.vestmarkets.com/v2"
ACCT_GRP  = 0
PAGE_SIZE = 500   # API hard cap (returns 500 even if you request more)

SYMBOLS    = ["BTC-PERP", "SOL-PERP"]
INTERVALS  = ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]

# Approximate ms per candle — used to detect when we've hit the beginning of history
INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
    "1M": 2_592_000_000,
}

# Coverage of 500 candles at each interval (approximate)
INTERVAL_COVERAGE = {
    "1m": "~8 hours", "5m": "~1.7 days", "15m": "~5 days",
    "30m": "~10 days", "1h": "~21 days", "4h": "~83 days",
    "1d": "~500 days (full history)",
}

# Exchange launch ≈ Oct 20 2024 UTC
EXCHANGE_GENESIS_MS = 1_729_382_400_000


def fetch_page(symbol: str, interval: str,
               end_ms: int = None, verbose: bool = False) -> list:
    """Fetch one page of klines, newest-first."""
    params = {"symbol": symbol, "interval": interval, "limit": PAGE_SIZE}
    if end_ms:
        params["endTime"] = end_ms

    url = f"{BASE_URL}/klines?" + parse.urlencode(params)
    req = request.Request(url, headers={
        "xrestservermm": f"restserver{ACCT_GRP}",
        "Accept": "application/json",
    })
    if verbose:
        print(f"    GET {url}")

    try:
        with request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
            rows = data["data"] if isinstance(data, dict) else data
            return rows
    except error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.reason}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return []


def raw_to_dict(row: list) -> dict:
    return {
        "open_time":   int(row[0]),
        "open":        float(row[1]),
        "high":        float(row[2]),
        "low":         float(row[3]),
        "close":       float(row[4]),
        "volume":      float(row[5]),
        "close_time":  int(row[6]),
    }


def ingest_symbol(db: MarketDB, symbol: str, interval: str,
                  incremental: bool = False, verbose: bool = False) -> int:
    """
    Paginate backwards through all available history for (symbol, interval).
    In incremental mode, only fetches candles newer than what's already stored.
    Returns total candles written.
    """
    label = f"{symbol}/{interval}"

    if incremental:
        # Only fetch from the latest stored candle forward
        latest_ms = db.latest_timestamp(symbol, interval)
        if latest_ms:
            # Re-fetch the latest candle (it might be a partial/in-progress one)
            # and fetch everything after it
            start_from = latest_ms
            print(f"  [{label}] incremental from {ms_to_dt(latest_ms)}")
        else:
            start_from = None
            print(f"  [{label}] no existing data — full fetch")
        return _ingest_forward(db, symbol, interval, since_ms=start_from, verbose=verbose)
    else:
        return _ingest_backward(db, symbol, interval, verbose=verbose)


def _ingest_backward(db: MarketDB, symbol: str, interval: str,
                     verbose: bool = False) -> int:
    """
    Fetch what the API provides for (symbol, interval).

    NOTE: The Vest API ignores startTime/endTime — it always returns the
    latest 500 candles. Pagination is not supported. We fetch once, store,
    and report what we got.
    """
    label = f"{symbol}/{interval}"

    rows = fetch_page(symbol, interval, verbose=verbose)
    if not rows:
        print(f"  [{label}] empty response — skipping")
        return 0

    dicts = [raw_to_dict(r) for r in rows]
    timestamps = sorted(d["open_time"] for d in dicts)
    oldest_ts  = timestamps[0]
    newest_ts  = timestamps[-1]

    written = db.upsert_candles(symbol, interval, dicts)
    print(f"  [{label}] {written} candles  {ms_to_dt(oldest_ts)} → {ms_to_dt(newest_ts)}")
    return written


def _ingest_forward(db: MarketDB, symbol: str, interval: str,
                    since_ms: int = None, verbose: bool = False) -> int:
    """Fetch only new candles (for incremental updates)."""
    label = f"{symbol}/{interval}"

    # Fetch latest page
    rows = fetch_page(symbol, interval, verbose=verbose)
    if not rows:
        return 0

    dicts = [raw_to_dict(r) for r in rows]

    if since_ms:
        # Only keep candles newer than what we already have
        dicts = [d for d in dicts if d["open_time"] >= since_ms]

    if not dicts:
        print(f"  [{label}] up to date")
        return 0

    written = db.upsert_candles(symbol, interval, dicts)
    newest  = ms_to_dt(max(d["open_time"] for d in dicts))
    print(f"  [{label}] wrote {written} new candles (latest: {newest})")
    return written


def ms_to_dt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def main():
    parser = argparse.ArgumentParser(description="Ingest Vest Markets klines into DuckDB")
    parser.add_argument("--db",          default=None,         help="DuckDB path")
    parser.add_argument("--symbols",     default=",".join(SYMBOLS),
                        help=f"Comma-separated symbols (default: {','.join(SYMBOLS)})")
    parser.add_argument("--intervals",   default=",".join(INTERVALS),
                        help=f"Comma-separated intervals (default: {','.join(INTERVALS)})")
    parser.add_argument("--incremental", action="store_true",
                        help="Only fetch candles newer than what's stored")
    parser.add_argument("--verbose",     action="store_true",  help="Show URLs")
    args = parser.parse_args()

    db_path   = Path(args.db) if args.db else None
    symbols   = [s.strip() for s in args.symbols.split(",")]
    intervals = [i.strip() for i in args.intervals.split(",")]

    db_kwargs = {"path": db_path} if db_path else {}
    db = MarketDB(**db_kwargs)

    mode = "incremental" if args.incremental else "full historical"
    print(f"=== Vest Markets Ingestion ({mode}) ===")
    print(f"DB:        {db.path}")
    print(f"Symbols:   {symbols}")
    print(f"Intervals: {intervals}")
    print()

    grand_total = 0
    for symbol in symbols:
        for interval in intervals:
            print(f"▶ {symbol} / {interval}")
            n = ingest_symbol(db, symbol, interval,
                              incremental=args.incremental,
                              verbose=args.verbose)
            grand_total += n
            print()

    print(f"=== Done: {grand_total} total candles written ===\n")
    print("Database summary:")
    db.summary()
    db.close()


if __name__ == "__main__":
    main()
