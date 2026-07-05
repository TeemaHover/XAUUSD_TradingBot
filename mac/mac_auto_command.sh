#!/bin/bash
# ============================================================
#  Full AI training pipeline - Mac/Linux
#  Run:  bash mac/mac_auto_command.sh
#  Steps: deps -> data (with backup) -> build -> train -> predict
#         -> HMM regimes -> backtest -> sweep -> per-regime report
# ============================================================
set -e
cd "$(dirname "$0")/.."

# dodger blue (30,144,255)
BLUE=$'\033[38;2;30;144;255m'
RESET=$'\033[0m'
step() { echo; echo "${BLUE}$1${RESET}"; }

step "[1/9] Checking dependencies..."
python3 --version
node --version
pip3 install numpy --quiet

step "[2/9] Downloading data (skipped if already present)..."
if [ ! -f data/gold_5m.csv ]; then
    python3 scripts/download_data.py 3
else
    echo "    data/gold_5m.csv already exists - skipping download"
    mkdir -p data/backup
    cp -f data/gold_5m.csv data/gold_1h.csv data/gold_4h.csv data/backup/ 2>/dev/null || true
    echo "    (backed up existing CSVs to data/backup/)"
fi

step "[3/9] Building TypeScript backtester..."
npm install --silent
npm run build

step "[4/9] Training baseline model (this takes a LONG time - hours on CPU)..."
python3 scripts/ai_train.py data/gold_5m.csv \
    --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
    --label-mode triple --tp-r 2 --sl-r 1 --horizon 96 --stride 3 \
    --features base,patterns --out models/m_baseline.npz

step "[5/9] Generating holdout predictions..."
python3 scripts/ai_backtest_predict.py data/gold_5m.csv \
    --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
    --model models/m_baseline.npz --embargo 96 --out data/preds_baseline.csv

step "[6/9] Fitting HMM market regimes..."
python3 scripts/hmm_regime.py data/gold_5m.csv

step "[7/9] Backtesting AI vs rules..."
LOG_LEVEL=warn node dist/src/backtest/runBacktest.js data/gold_5m.csv --ai data/preds_baseline.csv

step "[8/9] Confidence threshold sweep..."
LOG_LEVEL=warn node dist/src/backtest/runBacktest.js data/gold_5m.csv --ai data/preds_baseline.csv --sweep

step "[9/9] Per-regime breakdown (where does the money come from?)..."
python3 scripts/regime_report.py backtest-ai.json data/hmm_regimes.csv
python3 scripts/regime_report.py backtest-rules-holdout.json data/hmm_regimes.csv

echo
echo "${BLUE}============================================================${RESET}"
echo "${BLUE} DONE. Read in this order:${RESET}"
echo "${BLUE}   1. AI vs RULES table      - does AI beat rules? (expectancy)${RESET}"
echo "${BLUE}   2. Sweep table            - best confidence threshold${RESET}"
echo "${BLUE}   3. Per-regime tables      - is one market state losing money?${RESET}"
echo "${BLUE} Details: backtest-ai.json / backtest-rules-holdout.json${RESET}"
echo "${BLUE}============================================================${RESET}"
