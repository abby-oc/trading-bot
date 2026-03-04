#!/usr/bin/env python3
"""
pattern_calibrator.py — Offline calibrator for pattern win rates.

Walks historical SOL-PERP kline data to estimate per-pattern win rates
via forward-looking evaluation. Writes calibrated stats to data/pattern_stats.json.

Usage:
    python3 scripts/pattern_calibrator.py

Output:
    data/pattern_stats.json
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from market_db import MarketDB
from pattern_engine import PatternEngine

# ── Config ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.4    # Min composite confidence to evaluate
FORWARD_CANDLES      = 5      # How many 1m candles to look ahead
WIN_THRESHOLD_PCT    = 0.20   # % move in predicted direction = win
DATA_PATH            = PROJECT_DIR / "data" / "pattern_stats.json"

# Minimum window sizes before running detectors
MIN_1M  = 25
MIN_5M  = 35
MIN_15M = 45

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("calibrator")


def main():
    log.info("📊 Pattern Calibrator starting...")
    log.info(f"   Forward candles: {FORWARD_CANDLES}")
    log.info(f"   Win threshold: >{WIN_THRESHOLD_PCT}% move in predicted direction")
    log.info(f"   Min confidence to evaluate: {CONFIDENCE_THRESHOLD}")

    # ── Load historical data ──────────────────────────────────────────────
    db = MarketDB(read_only=True)
    log.info("Loading historical klines from DuckDB...")

    df_1m_all  = db.ohlcv("SOL-PERP", "1m",  limit=700)
    df_5m_all  = db.ohlcv("SOL-PERP", "5m",  limit=600)
    df_15m_all = db.ohlcv("SOL-PERP", "15m", limit=600)
    db.close()

    log.info(f"   1m rows: {len(df_1m_all)}")
    log.info(f"   5m rows: {len(df_5m_all)}")
    log.info(f"   15m rows: {len(df_15m_all)}")

    if len(df_1m_all) < MIN_1M + FORWARD_CANDLES:
        log.error("Not enough 1m data for calibration.")
        return

    # ── Build engine with blank stats ─────────────────────────────────────
    # Use a temp stats file so we don't overwrite existing during calibration
    engine = PatternEngine(stats_path=DATA_PATH)
    # Reset to fresh 0.5 defaults for calibration pass
    for name in engine.stats:
        engine.stats[name] = {
            "win_rate": 0.5,
            "avg_move_pct": 0.0,
            "sample_count": 0,
            "wins": 0,
        }

    # ── Walk forward through 1m timestamps ───────────────────────────────
    # Index range: MIN_1M .. (len-1-FORWARD_CANDLES)
    total_rows  = len(df_1m_all)
    end_idx     = total_rows - FORWARD_CANDLES - 1
    evaluations = 0
    skipped_conf = 0
    per_pattern_records: dict[str, list] = defaultdict(list)

    # Cache 5m and 15m open_times for fast filtering
    ts_5m  = df_5m_all["open_time"].values
    ts_15m = df_15m_all["open_time"].values

    log.info(f"Walking {end_idx - MIN_1M} timestamps...")

    for i in range(MIN_1M, end_idx + 1):
        # Build 1m slice up to and including index i
        df_1m = df_1m_all.iloc[max(0, i - MIN_1M): i + 1].reset_index(drop=True)
        if len(df_1m) < MIN_1M:
            continue

        t_ms = int(df_1m_all.iloc[i]["open_time"])

        # Build 5m slice: all rows with open_time <= t_ms, last MIN_5M rows
        mask5 = ts_5m <= t_ms
        df_5m_slice = df_5m_all[mask5].tail(MIN_5M).reset_index(drop=True)
        if len(df_5m_slice) < MIN_5M:
            continue

        # Build 15m slice: all rows with open_time <= t_ms, last MIN_15M rows
        mask15 = ts_15m <= t_ms
        df_15m_slice = df_15m_all[mask15].tail(MIN_15M).reset_index(drop=True)
        if len(df_15m_slice) < MIN_15M:
            continue

        # Run pattern engine (no trade data in historical mode)
        composite = engine.analyze(df_1m, df_5m_slice, df_15m_slice, [])

        if composite.direction == "NEUTRAL" or composite.confidence < CONFIDENCE_THRESHOLD:
            skipped_conf += 1
            continue

        if not composite.signals:
            continue

        # Entry price: close of current 1m candle
        entry_price = float(df_1m_all.iloc[i]["close"])
        if entry_price == 0:
            continue

        # Look-ahead: candles i+1 .. i+FORWARD_CANDLES
        future = df_1m_all.iloc[i + 1: i + 1 + FORWARD_CANDLES]
        if len(future) < FORWARD_CANDLES:
            continue

        # Evaluate win at candle i+FORWARD_CANDLES close
        exit_close = float(future.iloc[-1]["close"])
        move_pct = (exit_close - entry_price) / entry_price * 100

        if composite.direction == "BUY":
            directional_move = move_pct
        else:  # SELL
            directional_move = -move_pct

        won = directional_move > WIN_THRESHOLD_PCT

        # Record per-pattern result
        for sig in composite.signals:
            per_pattern_records[sig.name].append({
                "won": won,
                "move_pct": abs(directional_move),
                "raw_confidence": sig.raw_confidence,
            })

        evaluations += 1

        if evaluations % 50 == 0:
            log.info(f"   Evaluated {evaluations} setups so far (i={i}/{end_idx})")

    log.info(f"\nTotal setups evaluated: {evaluations}")
    log.info(f"Skipped (low confidence): {skipped_conf}")

    # ── Aggregate per-pattern stats ────────────────────────────────────────
    result_stats = {}

    log.info("\n── Per-Pattern Results ──────────────────────────────")
    log.info(f"{'Pattern':<28} {'Samples':>7} {'Wins':>6} {'WinRate':>8} {'AvgMove':>9}")
    log.info("─" * 64)

    for name in [
        "VelocityBurst",
        "CompressionBreakout",
        "TrendPullback",
        "MeanReversion",
        "OrderFlowPressure",
        "MultitimeframeAlignment",
    ]:
        records = per_pattern_records.get(name, [])
        sample_count = len(records)
        wins = sum(1 for r in records if r["won"])
        win_rate = wins / sample_count if sample_count > 0 else 0.5
        avg_move = sum(r["move_pct"] for r in records) / sample_count if sample_count > 0 else 0.0

        result_stats[name] = {
            "win_rate":     round(win_rate, 4),
            "avg_move_pct": round(avg_move, 4),
            "sample_count": sample_count,
            "wins":         wins,
        }

        log.info(
            f"{name:<28} {sample_count:>7} {wins:>6} {win_rate:>8.1%} {avg_move:>8.3f}%"
        )

    log.info("─" * 64)

    # ── Save stats ────────────────────────────────────────────────────────
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(result_stats, f, indent=2)

    log.info(f"\n✅ Pattern stats saved to {DATA_PATH}")

    # Summary
    avg_wr = sum(v["win_rate"] for v in result_stats.values()) / len(result_stats)
    log.info(f"   Average win rate across patterns: {avg_wr:.1%}")
    log.info(f"   Total pattern activations evaluated: {evaluations}")


if __name__ == "__main__":
    main()
