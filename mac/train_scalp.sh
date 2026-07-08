#!/bin/bash
# ============================================================
#  Scalp model experiment: train -> predict -> backtest -> compare
#  Run:  caffeinate -i bash mac/train_scalp.sh
#
#  Scalp profile: TP=1R, SL=1R, resolve within 24 bars (2 hours).
#  Decision rule: keep ONLY if holdout expectancy beats the
#  baseline's 0.105R. More trades is NOT the goal - edge is.
# ============================================================
set -e
cd "$(dirname "$0")/.."

BLUE=$'\033[38;2;30;144;255m'
RESET=$'\033[0m'
step() { echo; echo "${BLUE}$1${RESET}"; }

step "[1/4] Training scalp model (TP 1R / SL 1R / 2h horizon)..."
python3 scripts/ai_train.py data/gold_5m.csv \
    --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
    --label-mode triple --tp-r 1 --sl-r 1 --horizon 24 --stride 3 \
    --features base,patterns --out models/m_scalp.npz

step "[2/4] Generating holdout predictions..."
python3 scripts/ai_backtest_predict.py data/gold_5m.csv \
    --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
    --model models/m_scalp.npz --embargo 24 --out data/preds_scalp.csv

step "[3/4] Backtesting scalp model on holdout..."
npm run build >/dev/null
LOG_LEVEL=warn node dist/src/backtest/runBacktest.js data/gold_5m.csv --ai data/preds_scalp.csv
cp backtest-ai.json backtest-scalp.json

step "[4/4] Confidence sweep for the scalp model..."
LOG_LEVEL=warn node dist/src/backtest/runBacktest.js data/gold_5m.csv --ai data/preds_scalp.csv --sweep

echo
echo "${BLUE}============================================================${RESET}"
echo "${BLUE} VERDICT TIME. Baseline to beat: expectancy 0.105R (530 trades)${RESET}"
echo "${BLUE}  - Scalp expectancy HIGHER  -> worth considering (tell Claude)${RESET}"
echo "${BLUE}  - Scalp expectancy LOWER   -> baseline stays, experiment done${RESET}"
echo "${BLUE} Scalp results saved to backtest-scalp.json${RESET}"
echo "${BLUE}============================================================${RESET}"
