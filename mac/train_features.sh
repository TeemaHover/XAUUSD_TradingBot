#!/bin/bash
# ============================================================
#  Feature-combo experiment (TRAINING_GUIDE Step 5):
#  same setup as the winning baseline, plus S/R distance and
#  order-block proximity features. Goal: HIGHER EXPECTANCY,
#  not more trades. Baseline to beat: 0.105R.
#  Run:  caffeinate -i bash mac/train_features.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."

BLUE=$'\033[38;2;30;144;255m'
RESET=$'\033[0m'
step() { echo; echo "${BLUE}$1${RESET}"; }

step "[1/3] Training baseline + sr_dist + ob_dist features..."
python3 scripts/ai_train.py data/gold_5m.csv \
    --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
    --label-mode triple --tp-r 2 --sl-r 1 --horizon 96 --stride 3 \
    --features base,patterns,sr_dist,ob_dist --out models/m_srob.npz

step "[2/3] Generating holdout predictions..."
python3 scripts/ai_backtest_predict.py data/gold_5m.csv \
    --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
    --model models/m_srob.npz --embargo 96 --out data/preds_srob.csv

step "[3/3] Backtesting on holdout..."
npm run build >/dev/null
LOG_LEVEL=warn node dist/src/backtest/runBacktest.js data/gold_5m.csv --ai data/preds_srob.csv
cp backtest-ai.json backtest-srob.json

echo
echo "${BLUE}============================================================${RESET}"
echo "${BLUE} Baseline to beat: expectancy 0.105R, PF 1.24, maxDD 10.8%${RESET}"
echo "${BLUE} Keep m_srob ONLY if it wins on BOTH val loss AND expectancy.${RESET}"
echo "${BLUE} Results saved to backtest-srob.json${RESET}"
echo "${BLUE}============================================================${RESET}"
