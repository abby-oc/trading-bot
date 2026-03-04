#!/usr/bin/env python3
"""
server.py — FastAPI data server for the trading bot frontend.

Endpoints:
  GET /api/klines?symbol=BTC-PERP&interval=1d&limit=500
  GET /api/latest
  GET /api/symbols
  GET /             → serves frontend/index.html
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from market_db import MarketDB
from strategy  import STRATEGIES, RatioZScore, make_strategies
from backtest  import run_backtest
from risk      import RiskManager

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
DB_PATH      = Path(__file__).parent.parent / "data" / "market.duckdb"

app = FastAPI(title="Trading Bot API")

# Serve static frontend files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def get_db():
    return MarketDB(path=DB_PATH, read_only=True)


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/symbols")
def symbols():
    db = get_db()
    try:
        rows = db.con.execute(
            "SELECT DISTINCT symbol FROM klines ORDER BY symbol"
        ).fetchall()
        syms = [r[0] for r in rows] + ["BTCSOL"]
        return {"symbols": syms}
    finally:
        db.close()


@app.get("/api/klines")
def klines(
    symbol: str   = Query("BTC-PERP"),
    interval: str = Query("1d"),
    limit: int    = Query(500, ge=1, le=1000),
):
    db = get_db()
    try:
        df = db.ohlcv(symbol, interval, limit=limit)
        if df.empty:
            raise HTTPException(404, f"No data for {symbol}/{interval}")

        # Lightweight Charts wants { time, open, high, low, close, value }
        # time must be Unix seconds (not ms)
        records = []
        for _, row in df.iterrows():
            records.append({
                "time":   int(row["open_time"]) // 1000,
                "open":   round(float(row["open"]),  6),
                "high":   round(float(row["high"]),  6),
                "low":    round(float(row["low"]),   6),
                "close":  round(float(row["close"]), 6),
                "volume": round(float(row.get("volume", row.get("btc_volume", 0)) or 0), 6),
            })
        return JSONResponse({"symbol": symbol, "interval": interval, "candles": records})
    finally:
        db.close()


@app.get("/api/latest")
def latest():
    db = get_db()
    try:
        df = db.con.execute("SELECT * FROM latest_prices ORDER BY symbol, interval").df()
        result = []
        for _, row in df.iterrows():
            result.append({
                "symbol":   row["symbol"],
                "interval": row["interval"],
                "close":    round(float(row["close"]), 6),
                "dt":       str(row["dt"]),
            })
        # Add BTCSOL from daily
        btcsol = db.con.execute("""
            SELECT close FROM btcsol_klines WHERE interval='1d'
            ORDER BY open_time DESC LIMIT 1
        """).fetchone()
        if btcsol:
            result.append({"symbol": "BTCSOL", "interval": "1d", "close": round(btcsol[0], 4), "dt": ""})
        return {"prices": result}
    finally:
        db.close()


@app.get("/api/risk")
def risk_endpoint(
    symbol:   str   = Query("BTC-PERP"),
    interval: str   = Query("1d"),
    leverage: float = Query(1.0),
    capital:  float = Query(10_000),
):
    db = get_db()
    try:
        df = db.ohlcv(symbol, interval, limit=200)
        if df.empty:
            raise HTTPException(404, f"No data for {symbol}/{interval}")
        rm = RiskManager(capital=capital, risk_pct=0.01, max_leverage=leverage)
        result = rm.assess(df, leverage=leverage)
        result["symbol"]   = symbol
        result["interval"] = interval
        result["price"]    = float(df["close"].iloc[-1])
        return JSONResponse(result)
    finally:
        db.close()


@app.get("/api/signals")
def signals_endpoint(symbol: str = Query(None), interval: str = Query("1d")):
    """Current signal from every strategy for the requested symbol (or all)."""
    db = get_db()
    try:
        out = []
        strats = make_strategies(interval)
        for name, strat in strats.items():
            is_ratio = isinstance(strat, RatioZScore)
            syms = ["BTCSOL"] if is_ratio else (
                [symbol] if symbol and not is_ratio else ["BTC-PERP", "SOL-PERP"]
            )
            for sym in syms:
                df = db.ohlcv(sym, interval, limit=500)
                if df.empty:
                    continue
                sig = strat.current_signal(df, sym)
                out.append({
                    "strategy":  sig.strategy,
                    "symbol":    sig.symbol,
                    "direction": sig.direction,
                    "strength":  round(sig.strength, 3),
                    "price":     sig.price,
                    "reason":    sig.reason,
                })
        return JSONResponse({"signals": out})
    finally:
        db.close()


@app.get("/api/backtest")
def backtest_endpoint(
    strategy: str = Query("ema_cross"),
    symbol:   str = Query(None),
    interval: str = Query("1d"),
    capital:  float = Query(10_000),
):
    db = get_db()
    try:
        strats = make_strategies(interval)
        strat = strats.get(strategy) or STRATEGIES.get(strategy)
        if not strat:
            raise HTTPException(400, f"Unknown strategy: {strategy}")
        is_ratio = isinstance(strat, RatioZScore)
        sym  = "BTCSOL" if is_ratio else (symbol or "BTC-PERP")
        df   = db.ohlcv(sym, interval, limit=500)
        if df.empty:
            raise HTTPException(404, f"No data for {sym}/{interval}")
        result = run_backtest(df, strat, sym, initial_capital=capital)
        return JSONResponse(result)
    finally:
        db.close()


@app.get("/api/indicators")
def indicators_endpoint(
    symbol:   str = Query("BTC-PERP"),
    interval: str = Query("1d"),
    limit:    int = Query(500),
):
    """Return klines with common indicators pre-computed."""
    from indicators import add_all
    db = get_db()
    try:
        df = db.ohlcv(symbol, interval, limit=limit)
        if df.empty:
            raise HTTPException(404)
        df = add_all(df)
        # Keep only what the frontend needs
        cols = ["open_time", "open", "high", "low", "close", "volume",
                "ema_12", "ema_26", "ema_50", "rsi_14",
                "bb_upper", "bb_mid", "bb_lower",
                "macd", "macd_signal", "macd_hist"]
        df = df[[c for c in cols if c in df.columns]].copy()
        df["open_time"] = (df["open_time"] / 1000).astype(int)  # → seconds
        # Replace NaN/Inf with None for JSON compliance
        import math
        def clean(v):
            if v is None: return None
            try:
                if math.isnan(v) or math.isinf(v): return None
            except (TypeError, ValueError): pass
            return v
        records = [{k: clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]
        return JSONResponse({"symbol": symbol, "interval": interval, "data": records})
    finally:
        db.close()


# ── Orderly Live Data ─────────────────────────────────────────────────────

import httpx
import math as _math

ORDERLY_BASE = "https://api-evm.orderly.org"
# Map interval strings to Orderly TV resolutions
ORDERLY_RES_MAP = {
    "1m": "1", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "4h": "240", "1d": "1D",
}
# Map our symbol names to Orderly symbols
ORDERLY_SYM_MAP = {
    "SOL-PERP": "PERP_SOL_USDC",
    "BTC-PERP": "PERP_BTC_USDC",
    "ETH-PERP": "PERP_ETH_USDC",
}

_http_client = None

def _get_http():
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(timeout=10)
    return _http_client


@app.get("/api/live/klines")
def live_klines(
    symbol: str   = Query("SOL-PERP"),
    interval: str = Query("1m"),
    limit: int    = Query(200, ge=1, le=1000),
):
    """Fetch live klines from Orderly Network."""
    orderly_sym = ORDERLY_SYM_MAP.get(symbol)
    if not orderly_sym:
        raise HTTPException(400, f"Unknown symbol: {symbol}")

    resolution = ORDERLY_RES_MAP.get(interval, "1")

    import time as _time
    now = int(_time.time())
    # Estimate seconds per bar
    secs = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "4h": 14400, "1d": 86400}.get(interval, 60)
    from_ts = now - (limit * secs)

    try:
        r = _get_http().get(f"{ORDERLY_BASE}/v1/tv/history", params={
            "symbol": orderly_sym,
            "resolution": resolution,
            "from": from_ts,
            "to": now,
        })
        data = r.json()
    except Exception as e:
        raise HTTPException(502, f"Orderly API error: {e}")

    if "t" not in data or not data["t"]:
        raise HTTPException(404, f"No live data for {symbol}/{interval}")

    candles = []
    for i in range(len(data["t"])):
        candles.append({
            "time":   data["t"][i],
            "open":   data["o"][i],
            "high":   data["h"][i],
            "low":    data["l"][i],
            "close":  data["c"][i],
            "volume": data["v"][i] if data["v"][i] else 0,
        })

    return JSONResponse({
        "symbol": symbol,
        "interval": interval,
        "source": "orderly",
        "candles": candles
    })


@app.get("/api/live/price")
def live_price(symbol: str = Query("SOL-PERP")):
    """Fetch latest price from Orderly."""
    orderly_sym = ORDERLY_SYM_MAP.get(symbol)
    if not orderly_sym:
        raise HTTPException(400, f"Unknown symbol: {symbol}")

    try:
        r = _get_http().get(f"{ORDERLY_BASE}/v1/public/market_trades",
                            params={"symbol": orderly_sym, "limit": 1})
        data = r.json()
    except Exception as e:
        raise HTTPException(502, f"Orderly API error: {e}")

    if data.get("success") and data["data"]["rows"]:
        t = data["data"]["rows"][0]
        return JSONResponse({
            "symbol": symbol,
            "price": t["executed_price"],
            "side": t["side"],
            "qty": t["executed_quantity"],
            "timestamp": t["executed_timestamp"],
            "source": "orderly",
        })
    raise HTTPException(404, "No price data")


@app.get("/api/live/scalper")
def scalper_status():
    """Read scalper log for current status (last 20 lines)."""
    log_dir = Path(__file__).parent.parent / "logs"
    logs = sorted(log_dir.glob("scalper_*.log"), reverse=True)
    if not logs:
        return JSONResponse({"running": False, "lines": []})

    lines = logs[0].read_text().splitlines()[-20:]
    return JSONResponse({
        "running": True,
        "log_file": logs[0].name,
        "lines": lines,
    })


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=7433, reload=False, log_level="warning")
