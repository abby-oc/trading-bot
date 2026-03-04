"""
market_db.py — DuckDB interface for Vest Markets OHLCV data.

Schema:
  klines(symbol, interval, open_time, open, high, low, close, volume, close_time)
  Views: btcsol_klines, latest_prices

Usage:
  from market_db import MarketDB
  db = MarketDB()
  df = db.ohlcv('BTC-PERP', '1h', limit=200)
  price = db.latest_close('SOL-PERP', '1h')
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "market.duckdb"

INTERVALS_ORDERED = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "3d", "1w", "1M"]


class MarketDB:
    def __init__(self, path: Path = DB_PATH, read_only: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            self._init_schema()
        else:
            self._init_views()  # views are session-scoped, safe in read-only

    # ──────────────────────────────────────────────
    # Schema
    # ──────────────────────────────────────────────

    def _init_views(self):
        """Create session-scoped views (no persistent writes — safe for read-only)."""
        self.con.execute("""
            CREATE OR REPLACE TEMP VIEW btcsol_klines AS
            SELECT
                b.interval,
                b.open_time,
                epoch_ms(b.open_time)        AS dt,
                b.open  / s.open             AS open,
                b.high  / s.low              AS high,
                b.low   / s.high             AS low,
                b.close / s.close            AS close,
                b.volume                     AS btc_volume,
                s.volume                     AS sol_volume,
                b.close_time
            FROM klines b
            JOIN klines s
              ON  b.interval  = s.interval
              AND b.open_time = s.open_time
            WHERE b.symbol = 'BTC-PERP'
              AND s.symbol  = 'SOL-PERP'
        """)
        self.con.execute("""
            CREATE OR REPLACE TEMP VIEW latest_prices AS
            SELECT
                symbol,
                interval,
                epoch_ms(open_time)  AS dt,
                open, high, low, close, volume
            FROM klines
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY symbol, interval
                ORDER BY open_time DESC
            ) = 1
        """)

    def _init_schema(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS klines (
                symbol     VARCHAR  NOT NULL,
                interval   VARCHAR  NOT NULL,
                open_time  BIGINT   NOT NULL,   -- ms since epoch
                open       DOUBLE   NOT NULL,
                high       DOUBLE   NOT NULL,
                low        DOUBLE   NOT NULL,
                close      DOUBLE   NOT NULL,
                volume     DOUBLE   DEFAULT 0,
                close_time BIGINT   DEFAULT 0,
                PRIMARY KEY (symbol, interval, open_time)
            )
        """)

        # Synthetic BTC/SOL ratio — high/low use conservative bounds
        self.con.execute("""
            CREATE OR REPLACE VIEW btcsol_klines AS
            SELECT
                b.interval,
                b.open_time,
                epoch_ms(b.open_time)        AS dt,
                b.open  / s.open             AS open,
                b.high  / s.low              AS high,   -- max ratio: BTC up, SOL down
                b.low   / s.high             AS low,    -- min ratio: BTC down, SOL up
                b.close / s.close            AS close,
                b.volume                     AS btc_volume,
                s.volume                     AS sol_volume,
                b.close_time
            FROM klines b
            JOIN klines s
              ON  b.interval  = s.interval
              AND b.open_time = s.open_time
            WHERE b.symbol = 'BTC-PERP'
              AND s.symbol  = 'SOL-PERP'
        """)

        # Latest closed candle per (symbol, interval)
        self.con.execute("""
            CREATE OR REPLACE VIEW latest_prices AS
            SELECT
                symbol,
                interval,
                epoch_ms(open_time)  AS dt,
                open, high, low, close, volume
            FROM klines
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY symbol, interval
                ORDER BY open_time DESC
            ) = 1
        """)

    # ──────────────────────────────────────────────
    # Writes
    # ──────────────────────────────────────────────

    def upsert_candles(self, symbol: str, interval: str, rows: list[dict]) -> int:
        """
        Insert or replace candles.  rows = list of dicts with keys:
          open_time, open, high, low, close, volume, close_time
        Returns number of rows written.
        """
        if not rows:
            return 0

        df = pd.DataFrame(rows)
        df["symbol"]   = symbol
        df["interval"] = interval
        df = df[["symbol", "interval", "open_time", "open", "high",
                 "low", "close", "volume", "close_time"]]

        # DuckDB INSERT OR REPLACE via temp table
        self.con.execute("CREATE OR REPLACE TEMP TABLE _upsert AS SELECT * FROM df")
        self.con.execute("""
            INSERT OR REPLACE INTO klines
            SELECT * FROM _upsert
        """)
        return len(df)

    # ──────────────────────────────────────────────
    # Reads
    # ──────────────────────────────────────────────

    def ohlcv(self, symbol: str, interval: str, limit: int = 500,
              start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> pd.DataFrame:
        """
        Return OHLCV DataFrame sorted oldest→newest.
        symbol: 'BTC-PERP', 'SOL-PERP', or 'BTCSOL'
        """
        if symbol == "BTCSOL":
            base = "btcsol_klines"
            filters = [f"interval = '{interval}'"]
        else:
            base = "klines"
            filters = [f"symbol = '{symbol}'", f"interval = '{interval}'"]

        if start_ms:
            filters.append(f"open_time >= {start_ms}")
        if end_ms:
            filters.append(f"open_time <= {end_ms}")

        where = " AND ".join(filters)
        query = f"""
            SELECT * FROM (
                SELECT * FROM {base}
                WHERE {where}
                ORDER BY open_time DESC
                LIMIT {limit}
            ) sub
            ORDER BY open_time ASC
        """
        df = self.con.execute(query).df()
        if "dt" not in df.columns and "open_time" in df.columns:
            df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df

    def latest_close(self, symbol: str, interval: str) -> Optional[float]:
        """Return the most recent close price."""
        if symbol == "BTCSOL":
            row = self.con.execute(
                f"SELECT close FROM btcsol_klines WHERE interval='{interval}' "
                f"ORDER BY open_time DESC LIMIT 1"
            ).fetchone()
        else:
            row = self.con.execute(
                f"SELECT close FROM klines WHERE symbol='{symbol}' AND interval='{interval}' "
                f"ORDER BY open_time DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def latest_timestamp(self, symbol: str, interval: str) -> Optional[int]:
        """Return the most recent open_time (ms). Used for incremental updates."""
        if symbol == "BTCSOL":
            # Use BTC leg
            symbol = "BTC-PERP"
        row = self.con.execute(
            f"SELECT MAX(open_time) FROM klines "
            f"WHERE symbol='{symbol}' AND interval='{interval}'"
        ).fetchone()
        return row[0] if row and row[0] else None

    def earliest_timestamp(self, symbol: str, interval: str) -> Optional[int]:
        """Return the oldest open_time (ms)."""
        row = self.con.execute(
            f"SELECT MIN(open_time) FROM klines "
            f"WHERE symbol='{symbol}' AND interval='{interval}'"
        ).fetchone()
        return row[0] if row and row[0] else None

    def count(self, symbol: str = None, interval: str = None) -> pd.DataFrame:
        """Return row counts per (symbol, interval)."""
        where = ""
        if symbol:
            where += f" AND symbol='{symbol}'"
        if interval:
            where += f" AND interval='{interval}'"
        return self.con.execute(f"""
            SELECT symbol, interval,
                   COUNT(*)                               AS candles,
                   epoch_ms(MIN(open_time))               AS first_candle,
                   epoch_ms(MAX(open_time))               AS last_candle
            FROM klines
            WHERE 1=1 {where}
            GROUP BY symbol, interval
            ORDER BY symbol, interval
        """).df()

    def summary(self) -> None:
        """Print a human-readable summary of stored data."""
        df = self.count()
        if df.empty:
            print("Database is empty.")
            return
        print(df.to_string(index=False))

    # ──────────────────────────────────────────────
    # Analytics helpers (for strategy use)
    # ──────────────────────────────────────────────

    def sma(self, symbol: str, interval: str, period: int, limit: int = 500) -> pd.DataFrame:
        """Simple moving average on close."""
        df = self.ohlcv(symbol, interval, limit=limit + period)
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
        return df.iloc[period:].reset_index(drop=True)

    def returns(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """Log returns on close."""
        import numpy as np
        df = self.ohlcv(symbol, interval, limit=limit + 1)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["pct_return"] = df["close"].pct_change()
        return df.dropna().reset_index(drop=True)

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
