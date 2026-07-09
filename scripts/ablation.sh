#!/bin/bash
# ============================================================
#  FEATURE ABLATION: which features earn their place?
#
#  For every combo: train -> holdout predict -> backtest, then
#  collect val_loss + expectancy + winrate into one table.
#  Resumable: finished combos are skipped (delete
#  logs/ablation/backtest_<name>.json to redo one).
#
#  Run everything (many hours - use caffeinate!):
#      caffeinate -i bash scripts/ablation.sh
#  Run only some combos:
#      bash scripts/ablation.sh fvg round poi_all
#  Faster rough pass:
#      EPOCHS=80 caffeinate -i bash scripts/ablation.sh
#
#  Decision rule (same as TRAINING_GUIDE): a combo wins only if
#  it beats 'baseline' on BOTH val_loss AND expectancy - and by
#  a real margin, not a hair. Many combos vs one holdout means
#  a tiny "win" is luck.
# ============================================================
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1   # live progress in the terminal despite tee

EPOCHS="${EPOCHS:-300}"
DIR=logs/ablation
RESULTS="$DIR/results.csv"
mkdir -p "$DIR"

# name  features   (leave-one-out of the baseline, then add-one POI/context)
combos=(
  "baseline    base,patterns"
  "base_only   base"
  "no_volpress base,engulfing,pinbar,dbl_retest,compression"
  "no_engulf   base,vol_pressure,pinbar,dbl_retest,compression"
  "no_pinbar   base,vol_pressure,engulfing,dbl_retest,compression"
  "no_dblret   base,vol_pressure,engulfing,pinbar,compression"
  "no_compress base,vol_pressure,engulfing,pinbar,dbl_retest"
  "sr          base,patterns,sr_dist"
  "ob          base,patterns,ob_dist"
  "fvg         base,patterns,fvg_dist"
  "round       base,patterns,round_dist"
  "poi_all     base,patterns,poi"
)

npm run build >/dev/null
[ -f "$RESULTS" ] || echo "name,features,val_loss,trades,win_rate,profit_factor,expectancy,max_dd,total_return" > "$RESULTS"

for entry in "${combos[@]}"; do
  name=$(awk '{print $1}' <<<"$entry")
  feats=$(awk '{print $2}' <<<"$entry")

  if [ $# -gt 0 ]; then
    keep=no; for a in "$@"; do [ "$a" = "$name" ] && keep=yes; done
    [ "$keep" = yes ] || continue
  fi
  if [ -f "$DIR/backtest_$name.json" ]; then
    echo "== $name already done, skipping"; continue
  fi

  echo; echo "========== $name  ($feats) =========="
  python3 scripts/ai_train.py data/gold_5m.csv \
      --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
      --label-mode triple --tp-r 2 --sl-r 1 --horizon 96 --stride 3 \
      --epochs "$EPOCHS" --features "$feats" \
      --out "models/ab_$name.npz" 2>&1 | tee "$DIR/train_$name.log"

  python3 scripts/ai_backtest_predict.py data/gold_5m.csv \
      --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
      --model "models/ab_$name.npz" --embargo 96 \
      --out "data/preds_ab_$name.csv" 2>&1 | tee "$DIR/predict_$name.log"

  LOG_LEVEL=warn node dist/src/backtest/runBacktest.js data/gold_5m.csv \
      --ai "data/preds_ab_$name.csv" --ai-only > "$DIR/backtest_$name.log" 2>&1
  cp backtest-ai.json "$DIR/backtest_$name.json"

  # "Restored best weights (epoch N, val_loss=X)" -> X
  val_loss=$(grep -o 'val_loss=[0-9.]*' "$DIR/train_$name.log" | tail -1 | cut -d= -f2)
  python3 - "$name" "$feats" "$val_loss" "$DIR/backtest_$name.json" "$RESULTS" <<'PY'
import csv, json, sys
name, feats, vl, bt, out = sys.argv[1:6]
d = json.load(open(bt))
row = [name, feats, vl, d["totalTrades"], f'{d["winRate"]:.4f}',
       f'{d["profitFactor"]:.4f}', f'{d["expectancy"]:.4f}',
       f'{d["maxDrawdown"]:.4f}', f'{d["totalReturn"]:.4f}']
with open(out, "a", newline="") as f:
    csv.writer(f).writerow(row)   # quotes the comma-separated features field
print("  ->", " ".join(map(str, row)))
PY
done

echo; echo "============ RESULTS (best expectancy first) ============"
python3 - "$RESULTS" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
rows.sort(key=lambda r: float(r["expectancy"]), reverse=True)
print(f'{"name":<12}{"val_loss":>9}{"trades":>7}{"WR":>7}{"PF":>7}{"exp_R":>8}{"maxDD":>7}{"return":>8}')
for r in rows:
    print(f'{r["name"]:<12}{r["val_loss"]:>9}{r["trades"]:>7}'
          f'{float(r["win_rate"]):>7.3f}{float(r["profit_factor"]):>7.3f}'
          f'{float(r["expectancy"]):>8.4f}{float(r["max_dd"]):>7.3f}'
          f'{float(r["total_return"]):>8.3f}')
PY
echo
echo "Winner check: must beat 'baseline' on BOTH val_loss AND exp_R, clearly."
echo "Then sweep its confidence threshold (raises win rate, fewer trades):"
echo "  node dist/src/backtest/runBacktest.js data/gold_5m.csv --ai data/preds_ab_WINNER.csv --sweep"
