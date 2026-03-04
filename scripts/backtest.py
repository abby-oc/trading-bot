"""
backtest.py — Vectorized backtester for trading strategies.

Usage:
    python3 scripts/backtest.py --strategy ema_cross --symbol BTC-PERP --interval 1d
    python3 scripts/backtest.py --strategy rsi_mean_rev --symbol SOL-PERP
    python3 scripts/backtest.py --strategy ratio_zscore
    python3 scripts/backtest.py --all
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from market_db import MarketDB
from strategy  import STRATEGIES, RatioZScore


# ── Core backtester ───────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    strategy,
    symbol: str,
    initial_capital: float = 10_000.0,
    commission: float = 0.0005,   # 0.05% per side (Vest taker fee)
    slippage: float   = 0.0002,   # 0.02% slippage estimate
) -> dict:
    """
    Vectorized backtest. Returns dict with metrics + trade log + equity curve.

    Assumptions:
    - Trades open on next bar's open after signal
    - Full capital deployment per trade (no partial sizing yet)
    - Commission + slippage applied on entry and exit
    - No leverage
    """
    result = strategy.run(df)
    result = result.copy()
    result["next_open"] = result["open"].shift(-1)

    trades  = []
    equity  = [initial_capital]
    capital = initial_capital
    pos     = 0       # current position: +1, -1, 0
    entry_price = 0.0
    entry_idx   = 0

    for i in range(len(result) - 1):
        sig  = int(result["signal"].iloc[i])
        nxt  = result["next_open"].iloc[i]
        if pd.isna(nxt):
            equity.append(capital)
            continue

        cost_pct = commission + slippage   # round-trip half (applied each side)

        if pos == 0 and sig != 0:
            # Enter
            pos         = sig
            entry_price = nxt * (1 + cost_pct * sig)   # adverse fill
            entry_idx   = i + 1

        elif pos != 0 and (sig == 0 or sig == -pos):
            # Exit
            exit_price = nxt * (1 - cost_pct * pos)    # adverse fill
            pnl_pct    = pos * (exit_price / entry_price - 1)
            trade_pnl  = capital * pnl_pct
            capital   += trade_pnl

            trades.append({
                "entry_dt":    str(result.index[entry_idx] if hasattr(result.index[entry_idx], 'isoformat') else result["open_time"].iloc[entry_idx]),
                "exit_dt":     str(result.index[i+1] if hasattr(result.index[i+1], 'isoformat') else result["open_time"].iloc[i+1]),
                "direction":   "LONG" if pos == 1 else "SHORT",
                "entry_price": round(entry_price, 4),
                "exit_price":  round(exit_price, 4),
                "pnl_pct":     round(pnl_pct * 100, 3),
                "pnl_usd":     round(trade_pnl, 2),
                "bars_held":   i + 1 - entry_idx,
            })

            if sig != 0:
                # Immediate reversal
                pos         = sig
                entry_price = nxt * (1 + cost_pct * sig)
                entry_idx   = i + 1
            else:
                pos = 0

        equity.append(capital)

    # Trim equity to match result length
    equity = equity[:len(result)]

    # ── Metrics ──────────────────────────────────────────────────────────
    equity_s = pd.Series(equity)
    returns  = equity_s.pct_change().dropna()

    total_return = (capital - initial_capital) / initial_capital * 100
    n_days       = len(result)
    years        = n_days / 365.0

    # Annualized return
    ann_return = ((capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Sharpe (daily returns, annualized, assuming 0 risk-free rate)
    sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0

    # Max drawdown
    peak    = equity_s.cummax()
    dd      = (equity_s - peak) / peak
    max_dd  = dd.min() * 100

    # Win rate
    if trades:
        wins     = sum(1 for t in trades if t["pnl_pct"] > 0)
        win_rate = wins / len(trades) * 100
        avg_win  = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] > 0] or [0])
        avg_loss = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0] or [0])
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Buy & hold for comparison
    bh_start  = result["close"].iloc[0]
    bh_end    = result["close"].iloc[-1]
    bh_return = (bh_end / bh_start - 1) * 100

    metrics = {
        "strategy":       strategy.name,
        "symbol":         symbol,
        "n_bars":         n_days,
        "n_trades":       len(trades),
        "final_capital":  round(capital, 2),
        "total_return":   round(total_return, 2),
        "ann_return":     round(ann_return, 2),
        "bh_return":      round(bh_return, 2),
        "sharpe":         round(sharpe, 3),
        "max_drawdown":   round(max_dd, 2),
        "calmar":         round(calmar, 3),
        "win_rate":       round(win_rate, 1),
        "avg_win_pct":    round(avg_win, 3),
        "avg_loss_pct":   round(avg_loss, 3),
        "profit_factor":  round(profit_factor, 3),
    }

    return {
        "metrics":      metrics,
        "trades":       trades,
        "equity_curve": [round(e, 2) for e in equity],
        "timestamps":   result["open_time"].tolist() if "open_time" in result.columns else [],
    }


def print_report(result: dict):
    m = result["metrics"]
    trades = result["trades"]

    sep = "─" * 52
    print(f"\n{'═'*52}")
    print(f"  Strategy : {m['strategy']}")
    print(f"  Symbol   : {m['symbol']}")
    print(f"  Bars     : {m['n_bars']}  |  Trades: {m['n_trades']}")
    print(sep)
    print(f"  Total return   : {m['total_return']:>8.2f}%  "
          f"  (B&H: {m['bh_return']:.2f}%)")
    print(f"  Ann. return    : {m['ann_return']:>8.2f}%")
    print(f"  Sharpe ratio   : {m['sharpe']:>8.3f}")
    print(f"  Max drawdown   : {m['max_drawdown']:>8.2f}%")
    print(f"  Calmar ratio   : {m['calmar']:>8.3f}")
    print(sep)
    print(f"  Win rate       : {m['win_rate']:>8.1f}%")
    print(f"  Avg win        : {m['avg_win_pct']:>8.3f}%")
    print(f"  Avg loss       : {m['avg_loss_pct']:>8.3f}%")
    print(f"  Profit factor  : {m['profit_factor']:>8.3f}")
    print(f"{'═'*52}\n")

    if trades:
        print("  Last 5 trades:")
        for t in trades[-5:]:
            sign = "▲" if t["direction"] == "LONG" else "▼"
            pnl_sign = "+" if t["pnl_pct"] >= 0 else ""
            print(f"    {sign} {t['direction']:<5}  "
                  f"{t['entry_price']:>10,.2f} → {t['exit_price']:>10,.2f}  "
                  f"pnl: {pnl_sign}{t['pnl_pct']:.2f}%  "
                  f"({t['bars_held']} bars)")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest trading strategies")
    parser.add_argument("--strategy", default="ema_cross",
                        choices=list(STRATEGIES.keys()) + ["all"],
                        help="Strategy name or 'all'")
    parser.add_argument("--symbol",   default=None, help="Override symbol")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--capital",  type=float, default=10_000)
    parser.add_argument("--all",      action="store_true", help="Run all strategies")
    args = parser.parse_args()

    db = MarketDB(read_only=True)

    run_all = args.all or args.strategy == "all"
    targets = list(STRATEGIES.items()) if run_all else [(args.strategy, STRATEGIES[args.strategy])]

    for strat_name, strat in targets:
        # Determine symbol and data source
        if isinstance(strat, RatioZScore):
            sym = "BTCSOL"
            iv  = args.interval
            df  = db.ohlcv("BTCSOL", iv, limit=500)
        else:
            sym = args.symbol or ("BTC-PERP" if strat_name == "ema_cross" else "SOL-PERP")
            iv  = args.interval
            df  = db.ohlcv(sym, iv, limit=500)

        if df.empty:
            print(f"No data for {sym}/{iv}")
            continue

        print(f"Running {strat_name} on {sym}/{iv} ({len(df)} bars)...")
        result = run_backtest(df, strat, sym, initial_capital=args.capital)
        print_report(result)

    db.close()


if __name__ == "__main__":
    main()
