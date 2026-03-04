"""
pattern_engine.py — Multi-pattern probabilistic signal engine for SOL-PERP.

Six pattern detectors:
  1. VelocityBurst       — momentum surge on 1m with vol confirmation
  2. CompressionBreakout — Bollinger squeeze breakout on 5m
  3. TrendPullback       — pullback entry in 15m trend direction
  4. MeanReversion       — oversold/overbought z-score reversal on 5m
  5. OrderFlowPressure   — live trade-flow imbalance (60s window)
  6. MultitimeframeAlignment — all 3 TFs pointing same direction

PatternAggregator combines signals weighted by confidence × win_rate.
PatternEngine is the top-level class: loads stats, runs detectors, persists stats.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# indicators are imported from local scripts/
import sys
sys.path.insert(0, str(Path(__file__).parent))
import indicators as ind

logger = logging.getLogger(__name__)

# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class PatternSignal:
    name: str
    direction: str           # "BUY", "SELL", or "NEUTRAL"
    raw_confidence: float    # 0.0–1.0
    expected_move_pct: float
    features: dict           # raw values for logging/debugging


@dataclass
class CompositeSignal:
    direction: str           # "BUY", "SELL", or "NEUTRAL"
    confidence: float        # weighted composite 0.0–1.0
    consensus: float         # fraction of active signals agreeing
    signals: list            # list[PatternSignal]
    active_patterns: list    # list[str]
    win_rate_weighted: float # avg win rate of signals on dominant side


# ── Helpers ───────────────────────────────────────────────────────────────

def _safe(series: pd.Series, idx: int = -1, default: float = float("nan")) -> float:
    """Safely get element from Series by position."""
    try:
        val = series.iloc[idx]
        return float(val) if not (val != val) else default  # nan check
    except Exception:
        return default


def _ema_dir(series: pd.Series, fast: int, slow: int) -> int:
    """Return 1 if fast EMA > slow EMA, -1 if <, 0 if insufficient data."""
    if len(series) < slow + 5:
        return 0
    fast_val = _safe(ind.ema(series, fast))
    slow_val = _safe(ind.ema(series, slow))
    if fast_val != fast_val or slow_val != slow_val:
        return 0
    if fast_val > slow_val:
        return 1
    elif fast_val < slow_val:
        return -1
    return 0


# ── Pattern 1: VelocityBurst ──────────────────────────────────────────────

def detect_velocity_burst(df_1m: pd.DataFrame, recent_trades: list) -> Optional[PatternSignal]:
    """
    Momentum surge: price moved >0.3% over last 5 bars with consistent bodies
    and volume confirmation.
    """
    if df_1m is None or len(df_1m) < 20:
        return None

    df = df_1m.tail(20).copy().reset_index(drop=True)
    close = df["close"]
    volume = df.get("volume", pd.Series([1.0] * len(df)))

    last_close = _safe(close, -1)
    close_5ago = _safe(close, -6)

    if last_close != last_close or close_5ago != close_5ago or close_5ago == 0:
        return None

    velocity_pct = (last_close - close_5ago) / close_5ago * 100

    # Volume ratio
    last_vol = _safe(volume, -1)
    avg_vol = float(volume.mean()) if float(volume.mean()) > 0 else 1.0
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

    # Consistency: last 5 bars body direction
    last5 = df.tail(5)
    if velocity_pct > 0:
        consistent = (last5["close"] > last5["open"]).sum()
    else:
        consistent = (last5["close"] < last5["open"]).sum()
    consistency = consistent / 5

    # RSI-7
    rsi_series = ind.rsi(close, 7)
    rsi_val = _safe(rsi_series, -1)

    # Trigger checks
    if abs(velocity_pct) <= 0.3:
        return None
    if consistency < 0.6:
        return None
    if velocity_pct > 0 and rsi_val != rsi_val:
        return None
    if velocity_pct > 0 and rsi_val >= 72:
        return None
    if velocity_pct < 0 and rsi_val <= 28:
        return None

    direction = "BUY" if velocity_pct > 0 else "SELL"
    raw_conf = (
        min(1.0, abs(velocity_pct) / 1.0)
        * consistency
        * min(1.0, vol_ratio / 1.5)
    )
    raw_conf = max(0.0, min(1.0, raw_conf))

    return PatternSignal(
        name="VelocityBurst",
        direction=direction,
        raw_confidence=raw_conf,
        expected_move_pct=abs(velocity_pct) * 0.5,
        features={
            "velocity_pct": round(velocity_pct, 4),
            "vol_ratio": round(vol_ratio, 3),
            "consistency": round(consistency, 2),
            "rsi": round(rsi_val, 2) if rsi_val == rsi_val else None,
        },
    )


# ── Pattern 2: CompressionBreakout ───────────────────────────────────────

def detect_compression_breakout(df_5m: pd.DataFrame) -> Optional[PatternSignal]:
    """
    Bollinger squeeze that breaks out: tight bands → explosive move.
    """
    if df_5m is None or len(df_5m) < 30:
        return None

    df = df_5m.tail(30).copy().reset_index(drop=True)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    bb = ind.bollinger(close, 20, 2.0)
    upper = bb["upper"]
    mid = bb["mid"]
    lower = bb["lower"]

    # Bollinger width series
    bb_width = (upper - lower) / mid.replace(0, np.nan)

    current_width = _safe(bb_width, -1)
    if current_width != current_width:
        return None

    # Squeeze: current width < min of last 20 width values
    width_lookback = bb_width.iloc[-20:] if len(bb_width) >= 20 else bb_width
    min_width = float(width_lookback.min())
    is_squeeze = current_width <= min_width

    if not is_squeeze:
        return None

    last_close = _safe(close, -1)
    last_upper = _safe(upper, -1)
    last_lower = _safe(lower, -1)
    last_mid = _safe(mid, -1)

    if any(v != v for v in [last_close, last_upper, last_lower, last_mid]) or last_mid == 0:
        return None

    broke_above = last_close > last_upper
    broke_below = last_close < last_lower

    if not (broke_above or broke_below):
        return None

    if broke_above:
        direction = "BUY"
        breakout_pct = (last_close - last_upper) / last_mid * 100
    else:
        direction = "SELL"
        breakout_pct = (last_lower - last_close) / last_mid * 100

    # ATR ratio
    atr_series = ind.atr(high, low, close, 14)
    current_atr = _safe(atr_series, -1)
    atr_mean = float(atr_series.mean()) if float(atr_series.mean()) > 0 else 1.0
    atr_ratio = current_atr / atr_mean if atr_mean > 0 else 1.0

    raw_conf = min(1.0, abs(breakout_pct) * 50) * min(1.0, atr_ratio)
    raw_conf = max(0.0, min(1.0, raw_conf))

    return PatternSignal(
        name="CompressionBreakout",
        direction=direction,
        raw_confidence=raw_conf,
        expected_move_pct=abs(breakout_pct) * 2,
        features={
            "bb_width": round(current_width, 5),
            "is_squeeze": True,
            "breakout_pct": round(breakout_pct, 4),
            "atr_ratio": round(atr_ratio, 3),
        },
    )


# ── Pattern 3: TrendPullback ──────────────────────────────────────────────

def detect_trend_pullback(df_15m: pd.DataFrame, df_5m: pd.DataFrame) -> Optional[PatternSignal]:
    """
    Pullback entry in direction of 15m trend using 5m for timing.
    """
    if df_15m is None or len(df_15m) < 40:
        return None
    if df_5m is None or len(df_5m) < 20:
        return None

    # 15m trend: EMA 8 vs 21
    df15 = df_15m.tail(40).copy().reset_index(drop=True)
    close15 = df15["close"]
    ema8_15 = ind.ema(close15, 8)
    ema21_15 = ind.ema(close15, 21)

    last_ema8 = _safe(ema8_15, -1)
    last_ema21 = _safe(ema21_15, -1)

    if any(v != v for v in [last_ema8, last_ema21]) or last_ema21 == 0:
        return None

    if last_ema8 > last_ema21:
        trend_direction = 1
    elif last_ema8 < last_ema21:
        trend_direction = -1
    else:
        trend_direction = 0

    if trend_direction == 0:
        return None

    ema_separation_pct = abs(last_ema8 - last_ema21) / last_ema21 * 100
    if ema_separation_pct <= 0.2:
        return None

    # 5m: pullback and resumption
    df5 = df_5m.tail(20).copy().reset_index(drop=True)
    close5 = df5["close"]
    low5 = df5["low"]
    ema21_5 = ind.ema(close5, 21)

    last_ema21_5 = _safe(ema21_5, -1)
    if last_ema21_5 != last_ema21_5 or last_ema21_5 == 0:
        return None

    # Pulled back: any of last 5 candles had low within 0.5% of EMA21
    last5_lows = low5.iloc[-5:]
    ema21_last5 = ema21_5.iloc[-5:]
    proximity = abs(last5_lows.values - ema21_last5.values) / ema21_last5.values * 100
    pulled_back = bool((proximity <= 0.5).any())

    if not pulled_back:
        return None

    # Resuming: latest candle in trend direction
    last_open5 = _safe(df5["open"], -1)
    last_close5 = _safe(close5, -1)
    if trend_direction == 1:
        resuming = last_close5 > last_open5
    else:
        resuming = last_close5 < last_open5

    if not resuming:
        return None

    # RSI ok: 5m RSI-9 between 35–65
    rsi5 = ind.rsi(close5, 9)
    rsi5_val = _safe(rsi5, -1)
    rsi_ok = (rsi5_val == rsi5_val) and (35 <= rsi5_val <= 65)

    if not rsi_ok:
        return None

    direction = "BUY" if trend_direction == 1 else "SELL"
    raw_conf = min(1.0, ema_separation_pct / 1.0) * 0.7  # rsi_ok is guaranteed here
    raw_conf = max(0.0, min(1.0, raw_conf))

    return PatternSignal(
        name="TrendPullback",
        direction=direction,
        raw_confidence=raw_conf,
        expected_move_pct=ema_separation_pct * 0.5,
        features={
            "trend_direction": trend_direction,
            "ema_separation_pct": round(ema_separation_pct, 4),
            "pulled_back": pulled_back,
            "resuming": resuming,
            "rsi_ok": rsi_ok,
            "rsi5": round(rsi5_val, 2) if rsi5_val == rsi5_val else None,
        },
    )


# ── Pattern 4: MeanReversion ──────────────────────────────────────────────

def detect_mean_reversion(df_5m: pd.DataFrame) -> Optional[PatternSignal]:
    """
    Z-score extreme with RSI + Bollinger %B confluence for mean-reversion entry.
    """
    if df_5m is None or len(df_5m) < 30:
        return None

    df = df_5m.tail(30).copy().reset_index(drop=True)
    close = df["close"]

    # Z-score (window=20)
    zs = ind.zscore(close, 20)
    zs_val = _safe(zs, -1)
    if zs_val != zs_val:
        return None

    if abs(zs_val) <= 1.8:
        return None

    # RSI-9
    rsi_series = ind.rsi(close, 9)
    rsi_val = _safe(rsi_series, -1)

    # Bollinger %B (20-period)
    bb = ind.bollinger(close, 20, 2.0)
    pct_b = _safe(bb["pct_b"], -1)

    # Oversold
    if zs_val < 0:
        if not (rsi_val == rsi_val and rsi_val < 35 and pct_b == pct_b and pct_b < 0.1):
            return None
        direction = "BUY"
    else:
        # Overbought
        if not (rsi_val == rsi_val and rsi_val > 65 and pct_b == pct_b and pct_b > 0.9):
            return None
        direction = "SELL"

    raw_conf = min(1.0, (abs(zs_val) - 1.8) / 1.2)
    raw_conf = max(0.0, min(1.0, raw_conf))

    return PatternSignal(
        name="MeanReversion",
        direction=direction,
        raw_confidence=raw_conf,
        expected_move_pct=abs(zs_val) * 0.15,
        features={
            "zscore": round(zs_val, 3),
            "rsi": round(rsi_val, 2) if rsi_val == rsi_val else None,
            "bb_pct_b": round(pct_b, 3) if pct_b == pct_b else None,
            "distance_from_mean": round(abs(zs_val), 3),
        },
    )


# ── Pattern 5: OrderFlowPressure ──────────────────────────────────────────

def detect_order_flow_pressure(recent_trades: list) -> Optional[PatternSignal]:
    """
    Buy/sell imbalance in live trade flow over last 60 seconds.
    Requires recent_trades: list of dicts with keys:
      side, executed_quantity, executed_timestamp (ms)
    """
    if not recent_trades:
        return None

    now_ms = time.time() * 1000
    window_ms = 60_000  # 60 seconds

    trades = [
        t for t in recent_trades
        if (now_ms - t.get("executed_timestamp", 0)) <= window_ms
    ]

    if not trades:
        return None

    buy_volume = sum(t.get("executed_quantity", 0) for t in trades if t.get("side") == "BUY")
    sell_volume = sum(t.get("executed_quantity", 0) for t in trades if t.get("side") == "SELL")
    total_volume = buy_volume + sell_volume
    trade_count = len(trades)

    if total_volume == 0:
        return None

    imbalance = (buy_volume - sell_volume) / total_volume

    if abs(imbalance) <= 0.55 or trade_count < 5:
        return None

    # Large trade bias
    quantities = [t.get("executed_quantity", 0) for t in trades]
    avg_qty = sum(quantities) / len(quantities) if quantities else 0
    large_trades = [t for t in trades if t.get("executed_quantity", 0) > avg_qty * 2]
    large_buy = sum(1 for t in large_trades if t.get("side") == "BUY")
    large_sell = sum(1 for t in large_trades if t.get("side") == "SELL")
    large_trade_bias = "BUY" if large_buy >= large_sell else "SELL"

    direction = "BUY" if imbalance > 0 else "SELL"
    raw_conf = min(1.0, (abs(imbalance) - 0.55) / 0.45)
    raw_conf = max(0.0, min(1.0, raw_conf))

    return PatternSignal(
        name="OrderFlowPressure",
        direction=direction,
        raw_confidence=raw_conf,
        expected_move_pct=abs(imbalance) * 0.3,
        features={
            "imbalance": round(imbalance, 3),
            "buy_volume": round(buy_volume, 4),
            "sell_volume": round(sell_volume, 4),
            "trade_count": trade_count,
            "large_trade_bias": large_trade_bias,
        },
    )


# ── Pattern 6: MultitimeframeAlignment ───────────────────────────────────

def detect_multitimeframe_alignment(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
) -> Optional[PatternSignal]:
    """
    All three timeframes point the same direction = high-conviction entry.
    """
    if df_1m is None or len(df_1m) < 20:
        return None
    if df_5m is None or len(df_5m) < 30:
        return None
    if df_15m is None or len(df_15m) < 30:
        return None

    # 1m: EMA 5 vs 13
    dir_1m = _ema_dir(df_1m["close"].tail(20), 5, 13)
    # 5m: EMA 8 vs 21
    dir_5m = _ema_dir(df_5m["close"].tail(30), 8, 21)
    # 15m: EMA 8 vs 21
    dir_15m = _ema_dir(df_15m["close"].tail(30), 8, 21)

    if not (dir_1m == dir_5m == dir_15m and dir_1m != 0):
        return None

    alignment_score = 1.0  # all 3 agree

    # Momentum check: RSI-7 on 1m
    rsi_1m = ind.rsi(df_1m["close"].tail(20), 7)
    rsi_val = _safe(rsi_1m, -1)

    # Reject if RSI is in extreme zone (< 25 for up, > 75 for down)
    if rsi_val == rsi_val:
        if dir_1m == 1 and rsi_val > 75:
            return None
        if dir_1m == -1 and rsi_val < 25:
            return None

    direction = "BUY" if dir_1m == 1 else "SELL"
    rsi_adj = (1 - abs(rsi_val - 50) / 50 * 0.3) if rsi_val == rsi_val else 0.7
    raw_conf = 0.9 * rsi_adj
    raw_conf = max(0.0, min(1.0, raw_conf))

    return PatternSignal(
        name="MultitimeframeAlignment",
        direction=direction,
        raw_confidence=raw_conf,
        expected_move_pct=0.4,
        features={
            "dir_1m": dir_1m,
            "dir_5m": dir_5m,
            "dir_15m": dir_15m,
            "alignment_score": alignment_score,
            "rsi_1m": round(rsi_val, 2) if rsi_val == rsi_val else None,
        },
    )


# ── PatternAggregator ─────────────────────────────────────────────────────

class PatternAggregator:
    """
    Combines individual PatternSignals into a CompositeSignal.
    Weighting: confidence × historical win_rate.
    """

    def aggregate(self, signals: list, stats: dict) -> CompositeSignal:
        """
        signals: list[PatternSignal] (may be empty)
        stats:   {pattern_name: {win_rate, ...}}
        """
        if not signals:
            return CompositeSignal(
                direction="NEUTRAL",
                confidence=0.0,
                consensus=0.0,
                signals=[],
                active_patterns=[],
                win_rate_weighted=0.0,
            )

        votes = []
        weights = []
        win_rates = []

        for sig in signals:
            wr = stats.get(sig.name, {}).get("win_rate", 0.5)
            w = sig.raw_confidence * wr
            v = w * (1 if sig.direction == "BUY" else -1)
            votes.append(v)
            weights.append(w)
            win_rates.append(wr)

        total_weight = sum(weights)
        if total_weight == 0:
            return CompositeSignal(
                direction="NEUTRAL",
                confidence=0.0,
                consensus=0.0,
                signals=signals,
                active_patterns=[s.name for s in signals],
                win_rate_weighted=0.0,
            )

        weighted_direction = sum(votes)
        composite_confidence = abs(weighted_direction) / total_weight

        # Dominant direction
        dominant = "BUY" if weighted_direction > 0 else "SELL"

        # Consensus: fraction of non-neutral signals on dominant side
        agreeing = [s for s in signals if s.direction == dominant]
        consensus = len(agreeing) / len(signals)

        # Win-rate weighted average for signals on dominant side
        if agreeing:
            agree_weights = [
                signals[i].raw_confidence * win_rates[i]
                for i, s in enumerate(signals) if s.direction == dominant
            ]
            agree_wrs = [
                win_rates[i]
                for i, s in enumerate(signals) if s.direction == dominant
            ]
            denom = sum(agree_weights) or 1.0
            win_rate_weighted = sum(w * r for w, r in zip(agree_weights, agree_wrs)) / denom
        else:
            win_rate_weighted = 0.5

        return CompositeSignal(
            direction=dominant,
            confidence=composite_confidence,
            consensus=consensus,
            signals=signals,
            active_patterns=[s.name for s in signals],
            win_rate_weighted=win_rate_weighted,
        )


# ── PatternEngine ─────────────────────────────────────────────────────────

DEFAULT_STATS = {
    "VelocityBurst":          {"win_rate": 0.50, "avg_move_pct": 0.30, "sample_count": 0, "wins": 0},
    "CompressionBreakout":    {"win_rate": 0.50, "avg_move_pct": 0.35, "sample_count": 0, "wins": 0},
    "TrendPullback":          {"win_rate": 0.50, "avg_move_pct": 0.40, "sample_count": 0, "wins": 0},
    "MeanReversion":          {"win_rate": 0.50, "avg_move_pct": 0.25, "sample_count": 0, "wins": 0},
    "OrderFlowPressure":      {"win_rate": 0.50, "avg_move_pct": 0.20, "sample_count": 0, "wins": 0},
    "MultitimeframeAlignment":{"win_rate": 0.50, "avg_move_pct": 0.45, "sample_count": 0, "wins": 0},
}

_DATA_DIR = Path(__file__).parent.parent / "data"
_DEFAULT_STATS_PATH = _DATA_DIR / "pattern_stats.json"


class PatternEngine:
    def __init__(self, stats_path: Path = None):
        self.stats_path = Path(stats_path) if stats_path else _DEFAULT_STATS_PATH
        self.stats = self._load_stats()
        self.aggregator = PatternAggregator()
        self._updates_since_save = 0

    # ── Stats persistence ──────────────────────────────────────────────────

    def _load_stats(self) -> dict:
        if self.stats_path.exists():
            try:
                with open(self.stats_path) as f:
                    loaded = json.load(f)
                # Merge with defaults to ensure all keys exist
                merged = {}
                for name, defaults in DEFAULT_STATS.items():
                    merged[name] = {**defaults, **loaded.get(name, {})}
                logger.info(f"Loaded pattern stats from {self.stats_path}")
                return merged
            except Exception as e:
                logger.warning(f"Failed to load pattern stats: {e} — using defaults")
        return {k: dict(v) for k, v in DEFAULT_STATS.items()}

    def _save_stats(self):
        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_path, "w") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save pattern stats: {e}")

    # ── Main API ───────────────────────────────────────────────────────────

    def analyze(
        self,
        df_1m: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        recent_trades: list,
    ) -> CompositeSignal:
        """Run all 6 detectors and return composite signal."""
        raw_signals = [
            detect_velocity_burst(df_1m, recent_trades),
            detect_compression_breakout(df_5m),
            detect_trend_pullback(df_15m, df_5m),
            detect_mean_reversion(df_5m),
            detect_order_flow_pressure(recent_trades),
            detect_multitimeframe_alignment(df_1m, df_5m, df_15m),
        ]
        active = [s for s in raw_signals if s is not None]
        return self.aggregator.aggregate(active, self.stats)

    def update_stats(self, pattern_name: str, won: bool, move_pct: float):
        """
        Update in-memory stats from live trade results.
        Saves to JSON every 10 updates.
        """
        if pattern_name not in self.stats:
            self.stats[pattern_name] = dict(DEFAULT_STATS.get(pattern_name, {
                "win_rate": 0.50, "avg_move_pct": 0.0, "sample_count": 0, "wins": 0
            }))

        s = self.stats[pattern_name]
        s["sample_count"] = s.get("sample_count", 0) + 1
        if won:
            s["wins"] = s.get("wins", 0) + 1
        s["win_rate"] = s["wins"] / s["sample_count"]
        # Running avg move
        n = s["sample_count"]
        s["avg_move_pct"] = (s.get("avg_move_pct", 0.0) * (n - 1) + move_pct) / n

        self._updates_since_save += 1
        if self._updates_since_save >= 10:
            self._save_stats()
            self._updates_since_save = 0
