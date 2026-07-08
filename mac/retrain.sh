#!/bin/bash
# ============================================================
#  Periodic relearn: fresh data -> retrain -> validate -> report.
#  Run every 1-2 months:  caffeinate -i bash mac/retrain.sh
#
#  SAFE BY DESIGN: the new model is saved as m_candidate.npz and
#  is NOT deployed automatically. Deploy manually only if it
#  beats the live model's holdout numbers:
#      cp models/m_candidate.npz models/ai_model.npz
# ============================================================
set -e
cd "$(dirname "$0")/.."

BLUE=$'\033[38;2;30;144;255m'
RESET=$'\033[0m'
step() { echo; echo "${BLUE}$1${RESET}"; }

step "[1/5] Backing up current data and downloading fresh history..."
mkdir -p data/backup
cp -f data/gold_5m.csv data/gold_1h.csv data/gold_4h.csv data/backup/ 2>/dev/null || true
bash scripts/download_data.sh 3

step "[2/5] Retraining on fresh data (same recipe as the live model)..."
python3 scripts/ai_train.py data/gold_5m.csv \
    --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
    --label-mode triple --tp-r 2 --sl-r 1 --horizon 96 --stride 3 \
    --features base,patterns --out models/m_candidate.npz

step "[3/5] Holdout predictions..."
python3 scripts/ai_backtest_predict.py data/gold_5m.csv \
    --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
    --model models/m_candidate.npz --embargo 96 --out data/preds_candidate.csv

step "[4/5] Backtesting candidate on the new holdout..."
npm run build >/dev/null
LOG_LEVEL=warn node dist/src/backtest/runBacktest.js data/gold_5m.csv --ai data/preds_candidate.csv
cp backtest-ai.json backtest-candidate.json

step "[5/5] Confidence sweep..."
LOG_LEVEL=warn node dist/src/backtest/runBacktest.js data/gold_5m.csv --ai data/preds_candidate.csv --sweep

echo
echo "${BLUE}============================================================${RESET}"
echo "${BLUE} Candidate results: backtest-candidate.json${RESET}"
echo "${BLUE} Compare against the LIVE model's demo results before deploying.${RESET}"
echo "${BLUE} Deploy only if better:  cp models/m_candidate.npz models/ai_model.npz${RESET}"
echo "${BLUE} Then restart the bot.${RESET}"
echo "${BLUE}============================================================${RESET}"
