"""
indicators.py — Pure technical indicator functions.

All functions accept pandas Series or DataFrames and return Series.
No TA-lib dependency — pure pandas/numpy.
"""

import numpy as np
import pandas as pd


# ── Trend ─────────────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def dema(series: pd.Series, period: int) -> pd.Series:
    """Double EMA — reduces lag."""
    e = ema(series, period)
    return 2 * e - ema(e, period)


def macd(series: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9
         ) -> pd.DataFrame:
    """Returns DataFrame with columns: macd, signal, hist."""
    fast_ema   = ema(series, fast)
    slow_ema   = ema(series, slow)
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({
        "macd":   macd_line,
        "signal": signal_line,
        "hist":   macd_line - signal_line,
    })


# ── Momentum ──────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1/period, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stoch_rsi(series: pd.Series, rsi_period: int = 14,
              k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
    r  = rsi(series, rsi_period)
    lo = r.rolling(k_period).min()
    hi = r.rolling(k_period).max()
    k  = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    d  = sma(k, d_period)
    return pd.DataFrame({"k": k, "d": d})


# ── Volatility ────────────────────────────────────────────────────────────

def bollinger(series: pd.Series, period: int = 20,
              std_dev: float = 2.0) -> pd.DataFrame:
    mid  = sma(series, period)
    std  = series.rolling(period).std()
    return pd.DataFrame({
        "upper": mid + std_dev * std,
        "mid":   mid,
        "lower": mid - std_dev * std,
        "pct_b": (series - (mid - std_dev * std)) / (2 * std_dev * std),
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def historical_vol(series: pd.Series, period: int = 20,
                   annualize: int = 365) -> pd.Series:
    """Annualized historical volatility from log returns."""
    lr = np.log(series / series.shift(1))
    return lr.rolling(period).std() * np.sqrt(annualize)


# ── Statistical ───────────────────────────────────────────────────────────

def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - mean) / std over window."""
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / s.replace(0, np.nan)


def rolling_correlation(a: pd.Series, b: pd.Series,
                         window: int) -> pd.Series:
    return a.rolling(window).corr(b)


# ── Composite helpers ─────────────────────────────────────────────────────

def trend_strength(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Simple trend strength: slope of linear regression / ATR.
    Positive = uptrend, negative = downtrend.
    """
    def _slope(arr):
        x = np.arange(len(arr))
        if np.isnan(arr).any():
            return np.nan
        return np.polyfit(x, arr, 1)[0]

    slope = close.rolling(period).apply(_slope, raw=True)
    norm  = close.rolling(period).mean()
    return slope / norm * period  # normalize to % per period


def add_all(df: pd.DataFrame, close_col: str = "close",
            high_col: str = "high", low_col: str = "low") -> pd.DataFrame:
    """
    Convenience: add common indicators to a klines DataFrame.
    Returns new DataFrame with added columns.
    """
    c, h, l = df[close_col], df[high_col], df[low_col]
    out = df.copy()

    # Trend
    out["ema_12"]  = ema(c, 12)
    out["ema_26"]  = ema(c, 26)
    out["ema_50"]  = ema(c, 50)
    out["ema_200"] = ema(c, 200)
    out["sma_20"]  = sma(c, 20)

    # MACD
    _macd = macd(c)
    out["macd"]        = _macd["macd"]
    out["macd_signal"] = _macd["signal"]
    out["macd_hist"]   = _macd["hist"]

    # Momentum
    out["rsi_14"] = rsi(c, 14)

    # Volatility
    out["atr_14"] = atr(h, l, c, 14)
    _bb = bollinger(c, 20, 2)
    out["bb_upper"] = _bb["upper"]
    out["bb_mid"]   = _bb["mid"]
    out["bb_lower"] = _bb["lower"]
    out["bb_pct_b"] = _bb["pct_b"]
    out["hvol_20"]  = historical_vol(c, 20)

    # Log return
    out["log_return"] = np.log(c / c.shift(1))

    return out
