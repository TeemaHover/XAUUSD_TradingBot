#!/bin/bash
# Download free historical XAUUSD data from Dukascopy (no MT5 needed)
# and convert to the CSV format the AI trainer expects.
#
# Usage:  bash scripts/download_data.sh [years]   (default 3)
set -e
cd "$(dirname "$0")/.."

YEARS="${1:-3}"
FROM=$(python3 -c "from datetime import date,timedelta; print(date.today()-timedelta(days=int(365.25*$YEARS)))")
TO=$(python3 -c "from datetime import date; print(date.today())")
mkdir -p data/dl

download () {
  local tf="$1" out="$2"
  echo "==> Downloading XAUUSD $tf  ($FROM -> $TO) ..."
  rm -f data/dl/*.csv
  npx -y dukascopy-node -i xauusd -from "$FROM" -to "$TO" -t "$tf" -f csv -dir data/dl -bs 30 -bp 500
  local src
  src=$(ls -t data/dl/*.csv | head -1)
  # dukascopy-node writes: timestamp,open,high,low,close,volume (timestamp in ms)
  python3 - "$src" "$out" <<'EOF'
import csv, sys
from datetime import datetime, timezone
src, out = sys.argv[1], sys.argv[2]

def to_ms(v):
    v = v.strip()
    try:
        return int(float(v))  # already epoch ms
    except ValueError:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

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
EOF
}

download m5 data/gold_5m.csv
download h1 data/gold_1h.csv
download h4 data/gold_4h.csv

rm -rf data/dl
echo ""
echo "Done. Now train with:"
echo "  python3 scripts/ai_train.py data/gold_5m.csv --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv"
