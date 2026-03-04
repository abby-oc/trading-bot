"""
strategy.py — Strategy base class and concrete implementations.

Each strategy:
  - Takes a DataFrame of OHLCV data (with indicators pre-computed)
  - Returns a signal column: +1 = LONG, -1 = SHORT, 0 = FLAT
  - Exposes current_signal() for live use

Strategies:
  1. EMACross      — EMA 12/26 crossover filtered by RSI + MACD confirmation
  2. RSIMeanRev    — RSI oversold/overbought mean reversion
  3. RatioZScore   — BTC/SOL ratio z-score pairs mean reversion
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from indicators import add_all, ema, rsi, zscore, macd, bollinger


# ── Signal dataclass ──────────────────────────────────────────────────────

@dataclass
class Signal:
    strategy:  str
    symbol:    str
    direction: str          # "LONG" | "SHORT" | "FLAT"
    strength:  float        # 0.0 – 1.0
    price:     float
    reason:    str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata:  dict = field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        return self.direction != "FLAT"

    def __str__(self):
        icon = {"LONG": "▲", "SHORT": "▼", "FLAT": "●"}[self.direction]
        return (f"[{self.strategy}] {icon} {self.direction} {self.symbol} "
                f"@ {self.price:,.2f}  strength={self.strength:.2f}  "
                f"| {self.reason}")


# ── Base class ────────────────────────────────────────────────────────────

class Strategy(ABC):
    name: str = "base"
    default_interval: str = "1d"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators. Override to customize."""
        return add_all(df)

    @abstractmethod
    def signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute signal series from a prepared DataFrame.
        Returns pd.Series of int: +1 LONG, -1 SHORT, 0 FLAT.
        Index aligned to df.
        """

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with 'signal' and 'signal_strength' columns added."""
        prepared = self.prepare(df.copy())
        sig = self.signals(prepared)
        prepared["signal"] = sig.fillna(0).astype(int)
        prepared["signal_strength"] = self._strength(prepared)
        return prepared

    def _strength(self, df: pd.DataFrame) -> pd.Series:
        """Default strength = 1.0 wherever signal != 0, else 0. Override for nuance."""
        return (df["signal"] != 0).astype(float)

    def current_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Run strategy and return the most recent Signal object."""
        result = self.run(df)
        last   = result.iloc[-1]
        sig    = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT", 0: "FLAT"}[sig]
        strength  = float(last.get("signal_strength", 0))
        price     = float(last["close"])
        reason    = self._reason(result, -1)
        return Signal(
            strategy  = self.name,
            symbol    = symbol,
            direction = direction,
            strength  = strength,
            price     = price,
            reason    = reason,
        )

    def _reason(self, df: pd.DataFrame, idx: int) -> str:
        return ""


# ── Strategy 1: EMA Crossover ─────────────────────────────────────────────

class EMACross(Strategy):
    """
    EMA 12 / 26 crossover with three confirmation filters:
      1. MACD histogram > 0 (momentum confirms direction)
      2. RSI not in extreme zone (avoids chasing exhausted moves)
      3. Price above/below EMA-50 (higher-timeframe trend filter)

    Entry:  EMA12 crosses EMA26 with all filters aligned
    Exit:   Opposite crossover OR RSI extreme breach
    """
    name = "ema_cross"

    def __init__(self, fast: int = 12, slow: int = 26,
                 trend_ema: int = 50,
                 rsi_ob: float = 70, rsi_os: float = 30):
        self.fast     = fast
        self.slow     = slow
        self.trend    = trend_ema
        self.rsi_ob   = rsi_ob
        self.rsi_os   = rsi_os

    def signals(self, df: pd.DataFrame) -> pd.Series:
        fast    = df["ema_12"]
        slow    = df["ema_26"]
        trend   = df["ema_50"]
        rsi_s   = df["rsi_14"]
        hist    = df["macd_hist"]

        # Crossover detection
        cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        # Confirm: trend filter + MACD
        long_ok  = (df["close"] > trend) & (hist > 0) & (rsi_s < self.rsi_ob)
        short_ok = (df["close"] < trend) & (hist < 0) & (rsi_s > self.rsi_os)

        # Position state (hold until opposite cross)
        raw = pd.Series(0, index=df.index)
        raw[cross_up   & long_ok]  =  1
        raw[cross_down & short_ok] = -1

        # Forward-fill position (hold until exit)
        position = raw.replace(0, np.nan).ffill().fillna(0)

        # Hard exit: RSI extreme while already in position
        position[(position ==  1) & (rsi_s > 75)] = 0
        position[(position == -1) & (rsi_s < 25)] = 0

        return position.astype(int)

    def _strength(self, df: pd.DataFrame) -> pd.Series:
        """Strength = normalized distance between EMAs."""
        spread = (df["ema_12"] - df["ema_26"]).abs() / df["ema_26"]
        # Normalize to 0-1 over rolling 50-bar window
        norm = spread.rolling(50).rank(pct=True).fillna(0.5)
        return (df["signal"] != 0).astype(float) * norm

    def _reason(self, df: pd.DataFrame, idx: int) -> str:
        r = df.iloc[idx]
        return (f"EMA{self.fast}={r['ema_12']:.2f} vs EMA{self.slow}={r['ema_26']:.2f} | "
                f"RSI={r['rsi_14']:.1f} | MACD hist={r['macd_hist']:.2f}")


# ── Strategy 2: RSI Mean Reversion ────────────────────────────────────────

class RSIMeanRev(Strategy):
    """
    RSI oversold/overbought mean reversion.

    LONG  when RSI dips below oversold_entry and price is above BB lower band
    SHORT when RSI rises above overbought_entry and price is below BB upper band
    Exit  when RSI crosses back through the neutral zone (50 ± exit_band)

    The Bollinger Band filter avoids buying into free-fall or selling into
    strong breakouts where RSI can stay extreme for a long time.
    """
    name = "rsi_mean_rev"

    def __init__(self, period: int = 14,
                 ob_entry: float = 70, os_entry: float = 30,
                 ob_exit:  float = 55, os_exit:  float = 45):
        self.period   = period
        self.ob_entry = ob_entry
        self.os_entry = os_entry
        self.ob_exit  = ob_exit
        self.os_exit  = os_exit

    def signals(self, df: pd.DataFrame) -> pd.Series:
        rsi_s  = df["rsi_14"]
        close  = df["close"]
        bb_lo  = df["bb_lower"]
        bb_hi  = df["bb_upper"]

        position = pd.Series(0, index=df.index, dtype=int)
        pos = 0

        for i in range(1, len(df)):
            r    = rsi_s.iloc[i]
            c    = close.iloc[i]
            bbl  = bb_lo.iloc[i]
            bbh  = bb_hi.iloc[i]

            if pos == 0:
                # Entry
                if r < self.os_entry and c > bbl:   pos =  1
                elif r > self.ob_entry and c < bbh:  pos = -1
            elif pos == 1:
                # Long exit
                if r > self.os_exit:                  pos = 0
            elif pos == -1:
                # Short exit
                if r < self.ob_exit:                  pos = 0

            position.iloc[i] = pos

        return position

    def _strength(self, df: pd.DataFrame) -> pd.Series:
        """Strength = how far RSI is from the neutral zone."""
        rsi_s = df["rsi_14"]
        dist  = pd.Series(0.0, index=df.index)
        long_mask  = df["signal"] ==  1
        short_mask = df["signal"] == -1
        dist[long_mask]  = ((50 - rsi_s[long_mask])  / 50).clip(0, 1)
        dist[short_mask] = ((rsi_s[short_mask] - 50) / 50).clip(0, 1)
        return dist

    def _reason(self, df: pd.DataFrame, idx: int) -> str:
        r = df.iloc[idx]
        return (f"RSI={r['rsi_14']:.1f} | "
                f"BB pct-b={r['bb_pct_b']:.2f} | "
                f"close={r['close']:.2f}")


# ── Strategy 3: BTC/SOL Ratio Z-Score ─────────────────────────────────────

class RatioZScore(Strategy):
    """
    Pairs mean reversion on the synthetic BTC/SOL ratio.

    Trades the ratio expecting it to revert to a rolling mean:
      LONG  (BTC outperforms SOL) when ratio z-score < -entry_z  (ratio cheap)
      SHORT (SOL outperforms BTC) when ratio z-score >  entry_z  (ratio rich)
      Exit  when z-score reverts to ±exit_z

    Uses BTC/SOL daily close ratio derived from btcsol_klines view.
    """
    name             = "ratio_zscore"
    default_interval = "1d"

    def __init__(self, window: int = 60,
                 entry_z: float = 1.5, exit_z: float = 0.5):
        self.window  = window
        self.entry_z = entry_z
        self.exit_z  = exit_z

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """df must have 'close' = BTC/SOL ratio. Add z-score and ratio MA."""
        df["ratio"]        = df["close"]
        df["ratio_ma"]     = df["close"].rolling(self.window).mean()
        df["ratio_zscore"] = zscore(df["close"], self.window)
        df["ratio_vol"]    = df["close"].rolling(self.window).std()
        return df

    def signals(self, df: pd.DataFrame) -> pd.Series:
        z = df["ratio_zscore"]

        position = pd.Series(0, index=df.index, dtype=int)
        pos = 0

        for i in range(1, len(df)):
            zi = z.iloc[i]
            if np.isnan(zi):
                position.iloc[i] = 0
                continue

            if pos == 0:
                if zi < -self.entry_z:   pos =  1   # ratio too low → buy ratio (long BTC)
                elif zi >  self.entry_z: pos = -1   # ratio too high → sell ratio (long SOL)
            elif pos == 1:
                if zi > -self.exit_z:    pos = 0    # reversion complete
            elif pos == -1:
                if zi <  self.exit_z:    pos = 0

            position.iloc[i] = pos

        return position

    def _strength(self, df: pd.DataFrame) -> pd.Series:
        """Strength = |z-score| normalized, capped at 3σ."""
        return (df["ratio_zscore"].abs() / 3).clip(0, 1).fillna(0) * (df["signal"] != 0)

    def _reason(self, df: pd.DataFrame, idx: int) -> str:
        r = df.iloc[idx]
        return (f"BTC/SOL ratio={r['ratio']:.2f} | "
                f"z={r['ratio_zscore']:.2f} | "
                f"MA({self.window})={r['ratio_ma']:.2f}")


# ── Strategy registry ─────────────────────────────────────────────────────

STRATEGIES: dict[str, Strategy] = {
    "ema_cross":     EMACross(),
    "rsi_mean_rev":  RSIMeanRev(),
    "ratio_zscore":  RatioZScore(),
}
