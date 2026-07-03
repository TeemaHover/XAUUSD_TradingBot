"""
Download free historical XAUUSD data from Dukascopy (no MT5 needed) and
convert to the CSV format the AI trainer expects.

Cross-platform (Windows / Mac / Linux). Requires Node.js (for npx).

Usage:
    python scripts/download_data.py            # 3 years (default)
    python scripts/download_data.py 5          # 5 years
"""

import csv
import glob
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DL_DIR = os.path.join(ROOT, "data", "dl")

TIMEFRAMES = [("m5", "gold_5m.csv"), ("h1", "gold_1h.csv"), ("h4", "gold_4h.csv")]


def to_ms(v: str) -> int:
    v = v.strip()
    try:
        return int(float(v))  # already epoch ms
    except ValueError:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)


def convert(src: str, out: str) -> None:
    with open(src, newline="") as f, open(out, "w", newline="") as g:
        r = csv.DictReader(f)
        tcol = next(c for c in r.fieldnames if c.lower() in ("timestamp", "time", "date"))
        w = csv.writer(g)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        n = 0
        for row in r:
            w.writerow([to_ms(row[tcol]), row["open"], row["high"],
                        row["low"], row["close"], row.get("volume") or 1])
            n += 1
    print(f"    {n:,} candles -> {out}")


def download(tf: str, out_name: str, date_from: str, date_to: str) -> None:
    print(f"==> Downloading XAUUSD {tf}  ({date_from} -> {date_to}) ...")
    if os.path.isdir(DL_DIR):
        shutil.rmtree(DL_DIR)
    os.makedirs(DL_DIR, exist_ok=True)

    cmd = ["npx", "-y", "dukascopy-node", "-i", "xauusd",
           "-from", date_from, "-to", date_to, "-t", tf,
           "-f", "csv", "-dir", DL_DIR, "-bs", "30", "-bp", "500"]
    # shell=True so Windows finds npx.cmd on PATH
    result = subprocess.run(" ".join(cmd) if os.name == "nt" else cmd,
                            shell=(os.name == "nt"), cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"dukascopy-node failed for {tf} (is Node.js installed?)")

    files = sorted(glob.glob(os.path.join(DL_DIR, "*.csv")),
                   key=os.path.getmtime, reverse=True)
    if not files:
        sys.exit(f"No CSV produced for {tf}")
    convert(files[0], os.path.join(ROOT, "data", out_name))


def main() -> None:
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    date_from = str(date.today() - timedelta(days=int(365.25 * years)))
    date_to = str(date.today())

    for tf, out_name in TIMEFRAMES:
        download(tf, out_name, date_from, date_to)

    if os.path.isdir(DL_DIR):
        shutil.rmtree(DL_DIR)

    print("\nDone. Now train with:")
    print("  python scripts/ai_train.py data/gold_5m.csv "
          "--csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv "
          "--label-mode triple --tp-r 2 --sl-r 1 --horizon 96 --stride 3")


if __name__ == "__main__":
    main()
