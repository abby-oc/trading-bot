# trading-bot

Crypto trading bot targeting BTC, SOL, and the synthetic BTC/SOL ratio on [Vest Markets](https://vestmarkets.com).

## Stack

- **Data:** Vest Markets API (`/klines`) → DuckDB
- **Backend:** FastAPI + uvicorn
- **Frontend:** TradingView Lightweight Charts

## Structure

```
trading-bot/
├── data/               # DuckDB file (gitignored, rebuilt by ingest)
├── frontend/
│   └── index.html      # Chart UI
├── scripts/
│   ├── ingest.py       # Fetch & store historical klines
│   ├── market_db.py    # DuckDB read/write interface
│   └── server.py       # FastAPI data server
└── skills/
```

## Quickstart

```bash
pip install duckdb pandas fastapi uvicorn

# Ingest historical data
python3 scripts/ingest.py

# Start the server (http://localhost:7433)
python3 scripts/server.py
```

## Data

The Vest API returns the latest 500 candles per interval (no pagination):

| Interval | Coverage |
|----------|----------|
| `1d` | Full history since Oct 2024 (~500 days) |
| `4h` | Last ~83 days |
| `1h` | Last ~21 days |

Symbols: `BTC-PERP`, `SOL-PERP`, `BTCSOL` (synthetic ratio via DuckDB view).

## Incremental updates

```bash
python3 scripts/ingest.py --incremental
```
