import argparse
import logging
from dataclasses import dataclass

import pandas as pd

from strategy import build_signal


@dataclass
class BacktestTrade:
    time: pd.Timestamp
    action: str
    entry: float
    sl: float
    tp: float
    result: str
    pnl_r: float


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def resolve_trade(future_m5: pd.DataFrame, action: str, sl: float, tp: float, max_bars: int = 60):
    for _, candle in future_m5.head(max_bars).iterrows():
        if action == "BUY":
            if candle["low"] <= sl:
                return "SL", -1.0
            if candle["high"] >= tp:
                return "TP", 2.0
        else:
            if candle["high"] >= sl:
                return "SL", -1.0
            if candle["low"] <= tp:
                return "TP", 2.0
    return "TIMEOUT", 0.0


def run_backtest(m5: pd.DataFrame, h1: pd.DataFrame) -> list[BacktestTrade]:
    trades: list[BacktestTrade] = []

    for i in range(250, len(m5) - 61):
        now = m5.iloc[i]["time"]
        m5_window = m5.iloc[: i + 1]
        h1_window = h1[h1["time"] <= now].tail(300)
        if len(h1_window) < 210:
            continue

        signal = build_signal(m5_window, h1_window)
        if signal.action == "WAIT":
            logging.info("WAIT %s: %s", now, signal.reason)
            continue

        result, pnl_r = resolve_trade(
            m5.iloc[i + 1 :],
            signal.action,
            float(signal.sl),
            float(signal.tp),
        )
        logging.info(
            "%s %s: entry=%.2f sl=%.2f tp=%.2f result=%s",
            signal.action,
            now,
            signal.entry,
            signal.sl,
            signal.tp,
            result,
        )
        trades.append(
            BacktestTrade(
                time=now,
                action=signal.action,
                entry=float(signal.entry),
                sl=float(signal.sl),
                tp=float(signal.tp),
                result=result,
                pnl_r=pnl_r,
            )
        )

    return trades


def summarize(trades: list[BacktestTrade]) -> None:
    total = len(trades)
    wins = sum(1 for trade in trades if trade.result == "TP")
    losses = sum(1 for trade in trades if trade.result == "SL")
    pnl_r = sum(trade.pnl_r for trade in trades)
    win_rate = (wins / total * 100.0) if total else 0.0

    print(f"Trades: {total}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"PnL: {pnl_r:.2f}R")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the GOLD M5/H1 strategy.")
    parser.add_argument("--m5", required=True, help="CSV with M5 OHLC data")
    parser.add_argument("--h1", required=True, help="CSV with H1 OHLC data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    trades = run_backtest(load_csv(args.m5), load_csv(args.h1))
    summarize(trades)


if __name__ == "__main__":
    main()
