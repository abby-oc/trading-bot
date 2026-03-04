"""
risk.py — Volatility-aware risk management for short-term futures trading.

Key concerns for crypto perpetual futures:
  1. Volatility regime — are we in a calm period or about to get rekt?
  2. Position sizing — how much to put on given current vol + stop distance
  3. Liquidation distance — at X leverage, how far before margin call?
  4. Spike detection — sudden vol expansion signals danger

Usage:
  from risk import RiskManager
  rm = RiskManager(capital=10_000, risk_pct=0.01, max_leverage=5)
  regime = rm.regime(df)
  sizing = rm.position_size(entry=67_000, atr=1_400, leverage=3)
  liq    = rm.liquidation(entry=67_000, leverage=3, direction=1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from indicators import atr, historical_vol, zscore


# ── Volatility Regime ─────────────────────────────────────────────────────

class Regime(str, Enum):
    CALM     = "CALM"       # <25th percentile ATR  — tight stops, larger size
    NORMAL   = "NORMAL"     # 25–50th percentile    — standard params
    ELEVATED = "ELEVATED"   # 50–75th percentile    — reduce size
    HIGH     = "HIGH"       # 75–90th percentile    — small size, wider stops
    EXTREME  = "EXTREME"    # >90th percentile      — stand aside or micro size

    @property
    def color(self) -> str:
        return {
            "CALM":     "#3fb950",
            "NORMAL":   "#58a6ff",
            "ELEVATED": "#f0c040",
            "HIGH":     "#f08040",
            "EXTREME":  "#f85149",
        }[self.value]

    @property
    def size_multiplier(self) -> float:
        """Scale factor to apply to base position size."""
        return {
            "CALM":     1.25,
            "NORMAL":   1.00,
            "ELEVATED": 0.65,
            "HIGH":     0.35,
            "EXTREME":  0.10,
        }[self.value]

    @property
    def max_leverage(self) -> float:
        """Hard leverage cap for this regime."""
        return {
            "CALM":     10.0,
            "NORMAL":    5.0,
            "ELEVATED":  3.0,
            "HIGH":      2.0,
            "EXTREME":   1.0,
        }[self.value]


@dataclass
class RegimeSnapshot:
    regime:      Regime
    atr:         float          # current ATR in price units
    atr_pct:     float          # ATR as % of price
    atr_pctile:  float          # percentile rank (0–100) vs recent history
    hvol_20:     float          # 20-bar annualized hist vol (%)
    spike_score: float          # 0–1: how much recent vol exceeds recent baseline
    bars_in_regime: int         # consecutive bars at current regime level


@dataclass
class SizingResult:
    units:            float    # position size in base asset (e.g. BTC)
    notional:         float    # position value in quote (USDC)
    stop_distance:    float    # price distance to stop loss
    stop_price_long:  float    # stop loss if LONG
    stop_price_short: float    # stop loss if SHORT
    risk_amount:      float    # $ at risk (capital × risk_pct)
    stop_atr_mult:    float    # ATR multiples used for stop


@dataclass
class LiquidationResult:
    liq_price_long:   float   # liquidation if long
    liq_price_short:  float   # liquidation if short
    liq_distance_pct: float   # % move to liquidation (absolute)
    atr_to_liq:       float   # how many ATRs to liquidation
    is_dangerous:     bool    # True if liq_distance < 2×ATR
    warning:          str


class RiskManager:
    def __init__(
        self,
        capital:      float = 10_000,
        risk_pct:     float = 0.01,     # max 1% of capital per trade
        max_leverage: float = 5.0,
        stop_atr_mult: float = 2.0,     # stop = entry ± stop_atr_mult × ATR
        atr_period:   int   = 14,
        lookback:     int   = 100,      # bars for ATR percentile ranking
    ):
        self.capital       = capital
        self.risk_pct      = risk_pct
        self.max_leverage  = max_leverage
        self.stop_atr_mult = stop_atr_mult
        self.atr_period    = atr_period
        self.lookback      = lookback

    # ── Volatility Regime ─────────────────────────────────────────────────

    def regime_series(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of Regime values for every bar."""
        atr_s    = atr(df["high"], df["low"], df["close"], self.atr_period)
        atr_pct  = atr_s / df["close"]
        pctile   = atr_pct.rolling(self.lookback).rank(pct=True) * 100

        def _classify(p):
            if pd.isna(p): return Regime.NORMAL
            if p < 25:  return Regime.CALM
            if p < 50:  return Regime.NORMAL
            if p < 75:  return Regime.ELEVATED
            if p < 90:  return Regime.HIGH
            return Regime.EXTREME

        return pctile.map(_classify)

    def regime(self, df: pd.DataFrame) -> RegimeSnapshot:
        """Full regime snapshot for the most recent bar."""
        atr_s    = atr(df["high"], df["low"], df["close"], self.atr_period)
        atr_pct  = atr_s / df["close"]
        pctile   = atr_pct.rolling(self.lookback).rank(pct=True) * 100

        cur_atr     = float(atr_s.iloc[-1])
        cur_atr_pct = float(atr_pct.iloc[-1])
        cur_pctile  = float(pctile.iloc[-1]) if not pd.isna(pctile.iloc[-1]) else 50.0

        # 20-bar historical vol (annualized)
        hv = historical_vol(df["close"], 20)
        cur_hvol = float(hv.iloc[-1]) * 100 if not pd.isna(hv.iloc[-1]) else 0.0

        # Spike score: how much recent 5-bar ATR exceeds the 20-bar baseline
        atr_short = atr_s.rolling(5).mean()
        atr_base  = atr_s.rolling(20).mean()
        spike = float((atr_short / atr_base).iloc[-1]) if float(atr_base.iloc[-1]) > 0 else 1.0
        spike_score = min((spike - 1.0) / 2.0, 1.0) if spike > 1.0 else 0.0  # 0–1

        # Consecutive bars in current regime
        regime_s = self.regime_series(df)
        cur_regime = regime_s.iloc[-1]
        count = 0
        for r in reversed(regime_s.values):
            if r == cur_regime:
                count += 1
            else:
                break

        return RegimeSnapshot(
            regime         = cur_regime,
            atr            = round(cur_atr, 4),
            atr_pct        = round(cur_atr_pct * 100, 3),
            atr_pctile     = round(cur_pctile, 1),
            hvol_20        = round(cur_hvol, 1),
            spike_score    = round(spike_score, 3),
            bars_in_regime = count,
        )

    # ── Position Sizing ───────────────────────────────────────────────────

    def position_size(
        self,
        entry:         float,
        current_atr:   float,
        leverage:      float     = 1.0,
        regime:        Regime    = Regime.NORMAL,
        stop_atr_mult: float     = None,
    ) -> SizingResult:
        """
        ATR-based position sizing with regime scaling.

        Base size: risk_amount / stop_distance
        where stop_distance = stop_atr_mult × ATR
        and risk_amount = capital × risk_pct

        With leverage: notional can be larger, but units are still bounded
        by how much stop-loss we can absorb.
        """
        mult  = stop_atr_mult or self.stop_atr_mult
        scale = regime.size_multiplier
        lev   = min(leverage, regime.max_leverage, self.max_leverage)

        stop_dist   = mult * current_atr
        risk_amount = self.capital * self.risk_pct * scale
        units       = risk_amount / stop_dist if stop_dist > 0 else 0.0
        notional    = units * entry

        return SizingResult(
            units             = round(units, 6),
            notional          = round(notional, 2),
            stop_distance     = round(stop_dist, 4),
            stop_price_long   = round(entry - stop_dist, 4),
            stop_price_short  = round(entry + stop_dist, 4),
            risk_amount       = round(risk_amount, 2),
            stop_atr_mult     = mult,
        )

    # ── Liquidation Risk ──────────────────────────────────────────────────

    def liquidation(
        self,
        entry:        float,
        leverage:     float,
        current_atr:  float,
        maintenance_margin: float = 0.005,   # 0.5% typical for perps
    ) -> LiquidationResult:
        """
        Calculate liquidation prices and danger assessment.

        Isolated margin liquidation formula:
          Long:  liq = entry × (1 - 1/leverage + mm)
          Short: liq = entry × (1 + 1/leverage - mm)
        """
        if leverage <= 0:
            leverage = 1.0

        liq_long  = entry * (1 - 1/leverage + maintenance_margin)
        liq_short = entry * (1 + 1/leverage - maintenance_margin)

        dist_long  = (entry - liq_long)  / entry * 100
        dist_short = (liq_short - entry) / entry * 100
        dist_pct   = min(dist_long, dist_short)   # worst case

        atr_pct    = current_atr / entry * 100
        atr_to_liq = dist_pct / atr_pct if atr_pct > 0 else float("inf")

        is_dangerous = atr_to_liq < 2.0   # liq within 2 ATRs = danger zone

        if is_dangerous:
            warning = (f"⚠ LIQ in {atr_to_liq:.1f}×ATR — "
                       f"single spike could liquidate. Reduce leverage.")
        elif atr_to_liq < 4.0:
            warning = f"Caution: liq at {atr_to_liq:.1f}×ATR — volatility spike risk."
        else:
            warning = f"OK: liq at {atr_to_liq:.1f}×ATR from entry."

        return LiquidationResult(
            liq_price_long   = round(liq_long, 2),
            liq_price_short  = round(liq_short, 2),
            liq_distance_pct = round(dist_pct, 3),
            atr_to_liq       = round(atr_to_liq, 2),
            is_dangerous     = is_dangerous,
            warning          = warning,
        )

    # ── Full assessment ───────────────────────────────────────────────────

    def assess(
        self,
        df:       pd.DataFrame,
        leverage: float = 1.0,
    ) -> dict:
        """
        Full risk assessment for the current bar.
        Returns a flat dict suitable for JSON serialization.
        """
        snap    = self.regime(df)
        entry   = float(df["close"].iloc[-1])
        sizing  = self.position_size(entry, snap.atr, leverage, snap.regime)
        liq     = self.liquidation(entry, leverage, snap.atr)

        return {
            "regime":            snap.regime.value,
            "regime_color":      snap.regime.color,
            "atr":               snap.atr,
            "atr_pct":           snap.atr_pct,
            "atr_percentile":    snap.atr_pctile,
            "hvol_20":           snap.hvol_20,
            "spike_score":       snap.spike_score,
            "bars_in_regime":    snap.bars_in_regime,
            "size_multiplier":   snap.regime.size_multiplier,
            "max_leverage":      snap.regime.max_leverage,
            # Sizing
            "position_units":    sizing.units,
            "position_notional": sizing.notional,
            "stop_distance":     sizing.stop_distance,
            "stop_price_long":   sizing.stop_price_long,
            "stop_price_short":  sizing.stop_price_short,
            "risk_amount":       sizing.risk_amount,
            # Liquidation
            "liq_price_long":    liq.liq_price_long,
            "liq_price_short":   liq.liq_price_short,
            "liq_distance_pct":  liq.liq_distance_pct,
            "atr_to_liq":        liq.atr_to_liq,
            "liq_dangerous":     liq.is_dangerous,
            "liq_warning":       liq.warning,
        }
