"""
Resample a candle CSV to a higher timeframe.

You already have 6+ years of 5m/1h/4h gold data — no new download needed to
get daily (or 15m/30m) candles, just aggregate what's there:

    python scripts/resample_csv.py data/gold_1h.csv 1d  data/gold_1d.csv
    python scripts/resample_csv.py data/gold_5m.csv 15m data/gold_15m.csv
    python scripts/resample_csv.py data/gold_5m.csv 30m data/gold_30m.csv

Buckets are aligned to UTC. Output format matches the trainer/backtester:
time (epoch ms), open, high, low, close, volume.
"""

import csv
import sys

TF_MS = {
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, tf, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    if tf not in TF_MS:
        sys.exit(f"Unsupported timeframe {tf!r}. Choose from: {', '.join(TF_MS)}")
    tf_ms = TF_MS[tf]

    rows = []
    with open(src, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "time": int(float(row["time"])),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 1)),
            })
    if not rows:
        sys.exit(f"No candles in {src}")
    rows.sort(key=lambda r: r["time"])

    out = []
    current = None
    for r in rows:
        bucket = r["time"] // tf_ms * tf_ms
        if current is None or current["time"] != bucket:
            if current is not None:
                out.append(current)
            current = {"time": bucket, "open": r["open"], "high": r["high"],
                       "low": r["low"], "close": r["close"], "volume": r["volume"]}
        else:
            current["high"] = max(current["high"], r["high"])
            current["low"] = min(current["low"], r["low"])
            current["close"] = r["close"]
            current["volume"] += r["volume"]
    out.append(current)

    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        for c in out:
            w.writerow([c["time"], c["open"], c["high"], c["low"], c["close"], c["volume"]])

    print(f"{len(rows):,} {src} candles -> {len(out):,} {tf} candles -> {dst}")


if __name__ == "__main__":
    main()
