#!/bin/bash
# ============================================================
#  Full AI training pipeline - Mac/Linux
#  Run:  bash mac_auto_command.sh
#  Steps: deps -> data -> build -> train -> predict -> backtest -> sweep
# ============================================================
set -e
cd "$(dirname "$0")"

# dodger blue (30,144,255)
BLUE=$'\033[38;2;30;144;255m'
RESET=$'\033[0m'
step() { echo; echo "${BLUE}$1${RESET}"; }

step "[1/7] Checking dependencies..."
python3 --version
node --version
pip3 install numpy --quiet

step "[2/7] Downloading data (skipped if already present)..."
if [ ! -f data/gold_5m.csv ]; then
    python3 scripts/download_data.py 3
else
    echo "    data/gold_5m.csv already exists - skipping download"
fi

step "[3/7] Building TypeScript backtester..."
npm install --silent
npm run build

step "[4/7] Training baseline model (this takes a LONG time - hours on CPU)..."
python3 scripts/ai_train.py data/gold_5m.csv \
    --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
    --label-mode triple --tp-r 2 --sl-r 1 --horizon 96 --stride 3 \
    --features base,patterns --out models/m_baseline.npz

step "[5/7] Generating holdout predictions..."
python3 scripts/ai_backtest_predict.py data/gold_5m.csv \
    --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
    --model models/m_baseline.npz --embargo 96 --out data/preds_baseline.csv

step "[6/7] Backtesting AI vs rules..."
node dist/src/backtest/runBacktest.js data/gold_5m.csv --ai data/preds_baseline.csv

step "[7/7] Confidence threshold sweep..."
node dist/src/backtest/runBacktest.js data/gold_5m.csv --ai data/preds_baseline.csv --sweep

echo
echo "${BLUE}============================================================${RESET}"
echo "${BLUE} DONE. Compare expectancy in the tables above.${RESET}"
echo "${BLUE} Details: backtest-ai.json / backtest-rules-holdout.json${RESET}"
echo "${BLUE}============================================================${RESET}"
