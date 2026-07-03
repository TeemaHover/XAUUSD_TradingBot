"""
Export historical candles from your local MT5 terminal into the CSV format
the AI trainer expects. Alternative to download_data.py when Dukascopy is
unreachable. MT5 must be running and logged in.

Usage:
    python scripts/export_mt5_data.py            # symbol GOLD, 3 years
    python scripts/export_mt5_data.py XAUUSD 5   # custom symbol / years

Note: how much history you get depends on your broker's servers. The script
prints how many candles each timeframe actually returned — if 5m comes back
with much less than requested, increase "Max bars in chart" in MT5:
Tools > Options > Charts, set to Unlimited, restart MT5, run again.
"""

import csv
import os
import sys
from datetime import datetime, timedelta, timezone

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit("MetaTrader5 package missing. Run: pip install MetaTrader5")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TIMEFRAMES = [
    ("5m", None, "gold_5m.csv"),
    ("1h", None, "gold_1h.csv"),
    ("4h", None, "gold_4h.csv"),
]


# days per request, per timeframe — one giant range makes MT5 return
# "Invalid params", so we fetch in chunks and stitch them together
CHUNK_DAYS = {"5m": 21, "1h": 365, "4h": 1500}


def export(symbol: str, tf_name: str, mt5_tf, out_name: str,
           dt_from: datetime, dt_to: datetime) -> int:
    chunk = timedelta(days=CHUNK_DAYS.get(tf_name, 30))
    rows = {}
    cursor = dt_from
    failed_chunks = 0
    while cursor < dt_to:
        chunk_end = min(cursor + chunk, dt_to)
        rates = mt5.copy_rates_range(symbol, mt5_tf, cursor, chunk_end)
        if rates is None:
            failed_chunks += 1
        else:
            for r in rates:
                t = int(r["time"])
                rows[t] = [t * 1000, float(r["open"]), float(r["high"]),
                           float(r["low"]), float(r["close"]), float(r["tick_volume"])]
        cursor = chunk_end

    if not rows:
        print(f"  {tf_name}: NO DATA ({mt5.last_error()})")
        return 0

    ordered = [rows[t] for t in sorted(rows)]
    out = os.path.join(ROOT, "data", out_name)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        w.writerows(ordered)
    first = datetime.fromtimestamp(ordered[0][0] // 1000, tz=timezone.utc).date()
    last = datetime.fromtimestamp(ordered[-1][0] // 1000, tz=timezone.utc).date()
    note = f" ({failed_chunks} empty chunks - broker history limit)" if failed_chunks else ""
    print(f"  {tf_name}: {len(ordered):,} candles ({first} -> {last}) -> {out}{note}")
    return len(ordered)


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "GOLD"
    years = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    if not mt5.initialize():
        sys.exit(f"MT5 initialize failed: {mt5.last_error()} "
                 f"(is the MT5 terminal running and logged in?)")
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            sys.exit(f"Symbol '{symbol}' not found in MT5. "
                     f"Try: python scripts/export_mt5_data.py XAUUSD")
        if not info.visible:
            mt5.symbol_select(symbol, True)

        dt_to = datetime.now(timezone.utc)
        dt_from = dt_to - timedelta(days=int(365.25 * years))
        os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)

        print(f"Exporting {symbol}, {years} years ({dt_from.date()} -> {dt_to.date()}):")
        tf_map = {"5m": mt5.TIMEFRAME_M5, "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4}
        counts = {}
        for tf_name, _, out_name in TIMEFRAMES:
            counts[tf_name] = export(symbol, tf_name, tf_map[tf_name], out_name,
                                     dt_from, dt_to)

        expected_5m = int(years * 252 * 288 * 0.9)  # rough trading-time estimate
        if counts.get("5m", 0) < expected_5m * 0.5:
            print(f"\nWARNING: 5m returned {counts.get('5m', 0):,} candles, "
                  f"expected roughly {expected_5m:,}.")
            print("Your broker may limit history. In MT5: Tools > Options > Charts,")
            print("set 'Max bars in chart' to Unlimited, restart MT5, run again.")
            print("Training still works, just on a shorter period.")

        print("\nDone. Now run windows_auto_command.bat again "
              "(it will skip the download step).")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
