"""
Convert dukascopy-node CSV output to the bot's candle CSV format.

dukascopy-node columns:  timestamp,open,high,low,close,volume
bot format:              time,open,high,low,close,volume   (time = ms epoch)

Usage:
    python scripts/convert_dukascopy.py download/xauusd-m5-*.csv data/gold_5m.csv
    python scripts/convert_dukascopy.py download/xauusd-h1-*.csv data/gold_1h.csv
    python scripts/convert_dukascopy.py download/xauusd-h4-*.csv data/gold_4h.csv

Accepts multiple input files (glob) — they are merged, deduplicated by
timestamp, and sorted chronologically. Rows with zero/empty prices
(market closed) are dropped.
"""

import sys, csv, glob


def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: python scripts/convert_dukascopy.py <input.csv ...> <output.csv>")
    out_path = sys.argv[-1]
    in_paths = []
    for pattern in sys.argv[1:-1]:
        in_paths.extend(glob.glob(pattern))
    if not in_paths:
        sys.exit(f"No input files match: {sys.argv[1:-1]}")

    candles = {}
    for path in in_paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in reader.fieldnames}
            tcol = cols.get("timestamp") or cols.get("time")
            if tcol is None:
                sys.exit(f"{path}: no 'timestamp' or 'time' column (found {reader.fieldnames})")
            vcol = cols.get("volume")
            n = 0
            for row in reader:
                try:
                    t = int(float(row[tcol]))
                    o = float(row[cols["open"]])
                    h = float(row[cols["high"]])
                    l = float(row[cols["low"]])
                    c = float(row[cols["close"]])
                except (ValueError, KeyError):
                    continue
                if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                    continue  # market-closed placeholder rows
                v = 1.0
                if vcol and row.get(vcol):
                    try:
                        v = float(row[vcol]) or 1.0
                    except ValueError:
                        pass
                candles[t] = [t, o, h, l, c, v]
                n += 1
            print(f"{path}: {n:,} rows")

    rows = [candles[t] for t in sorted(candles)]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        w.writerows(rows)

    from datetime import datetime, timezone
    a = datetime.fromtimestamp(rows[0][0] / 1000, tz=timezone.utc)
    b = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc)
    print(f"\nWrote {len(rows):,} candles -> {out_path}")
    print(f"Range: {a:%Y-%m-%d} -> {b:%Y-%m-%d}")


if __name__ == "__main__":
    main()
