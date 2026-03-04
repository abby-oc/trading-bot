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
from strategy  import STRATEGIES, RatioZScore
from backtest  import run_backtest

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


@app.get("/api/signals")
def signals_endpoint(symbol: str = Query(None), interval: str = Query("1d")):
    """Current signal from every strategy for the requested symbol (or all)."""
    db = get_db()
    try:
        out = []
        for name, strat in STRATEGIES.items():
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
        strat = STRATEGIES.get(strategy)
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


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=7433, reload=False, log_level="warning")
