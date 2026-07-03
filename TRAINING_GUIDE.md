# AI Training Guide — Step by Step

Every command, in order. Run all commands from the TradingBot folder.
Windows: use `python` instead of `python3`.

---

## Step 0 — One-time setup

```bash
# download 3 years of XAUUSD 5m/1h/4h data from Dukascopy (free, no MT5 needed)
bash scripts/download_data.sh 3

# compile the TypeScript backtester
npm run build
```

Check: `data/gold_5m.csv`, `data/gold_1h.csv`, `data/gold_4h.csv` exist.

---

## Step 1 — Train the BASELINE model

Triple-barrier labels (model learns "does TP hit before SL", which is how the
bot actually trades). TP = 2xATR, SL = 1xATR, max 96 bars (8h).

```bash
python3 scripts/ai_train.py data/gold_5m.csv --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
  --label-mode triple --tp-r 2 --sl-r 1 --horizon 96 --stride 3 \
  --features base,patterns --out models/m_baseline.npz
```

At the end it prints **Validation accuracy** and a confusion matrix.
WRITE DOWN the validation loss and accuracy — every combo must beat this.

---

## Step 2 — Generate holdout predictions

`--embargo 96` MUST equal the `--horizon` used in training.

```bash
python3 scripts/ai_backtest_predict.py data/gold_5m.csv --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
  --model models/m_baseline.npz --embargo 96 --out data/preds_baseline.csv
```

---

## Step 3 — Backtest AI vs rules (same window, same costs)

```bash
node dist/src/backtest/runBacktest.js data/gold_5m.csv --ai data/preds_baseline.csv
```

Judge by **expectancy (avg R/trade)** and **maxDD**, not win rate.
WRITE DOWN the expectancy — this is the baseline number to beat.

---

## Step 4 — Confidence threshold sweep

```bash
node dist/src/backtest/runBacktest.js data/gold_5m.csv --ai data/preds_baseline.csv --sweep
```

Pick the threshold with the best expectancy that still has >50 trades.
Put it in `config/default.json` -> `strategy.aiConfidenceThreshold`.

---

## Step 5 — Test feature combinations (one at a time!)

Change ONLY `--features` and `--out`. Everything else stays identical.

```bash
# combo A: all 17 features
python3 scripts/ai_train.py data/gold_5m.csv --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
  --label-mode triple --tp-r 2 --sl-r 1 --horizon 96 --stride 3 \
  --features all --out models/m_all.npz

# combo B: + support/resistance and order-block proximity
python3 scripts/ai_train.py data/gold_5m.csv --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
  --label-mode triple --tp-r 2 --sl-r 1 --horizon 96 --stride 3 \
  --features base,patterns,sr_dist,ob_dist --out models/m_srob.npz

# combo C: + trendline geometry (wedges, triangles, rectangles)
python3 scripts/ai_train.py data/gold_5m.csv --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
  --label-mode triple --tp-r 2 --sl-r 1 --horizon 96 --stride 3 \
  --features base,patterns,slope_high,slope_low,convergence --out models/m_slopes.npz
```

Then repeat Step 2 + Step 3 for each combo (change `--model` and `--out`):

```bash
python3 scripts/ai_backtest_predict.py data/gold_5m.csv --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv \
  --model models/m_all.npz --embargo 96 --out data/preds_all.csv
node dist/src/backtest/runBacktest.js data/gold_5m.csv --ai data/preds_all.csv
```

### Decision rule
Keep a combo ONLY if it beats the baseline on BOTH:
1. validation loss (printed at end of training), AND
2. holdout expectancy (from Step 3)

If it only wins on one, it's probably noise. When in doubt, keep fewer features.

---

## Step 6 — Deploy the winner

```bash
cp models/m_WINNER.npz models/ai_model.npz
```

The model remembers its own feature list — live prediction and the MT5 bridge
extract the right features automatically. Set the threshold from Step 4 in
`config/default.json`, set `"aiMode": true`, and run on DEMO first.

---

## Extra commands

```bash
# rules engine: sniper zone entries vs market entries
node dist/src/backtest/runBacktest.js data/gold_5m.csv --compare-entries

# walk-forward test of the rules engine
node dist/src/backtest/runBacktest.js data/gold_5m.csv --walk-forward

# faster experiment iterations (rougher results)
#   add to any training command:  --epochs 100
# old-style labels (25-min direction) instead of triple-barrier:
#   --label-mode move --forward 5     (then use --embargo 5 in Step 2)
```

## Feature reference

| Group | Features | What it captures |
|---|---|---|
| base (7) | open_ret, body_ret, upper_wick, lower_wick, range_ret, vol_ratio, pos_20 | raw candle geometry + volume |
| patterns (5) | vol_pressure, engulfing, pinbar, dbl_retest, compression | candle patterns, double top/bottom, squeeze |
| context (5) | sr_dist, ob_dist, slope_high, slope_low, convergence | S/R touch, OB touch, wedges/triangles/rectangles |

## Warnings

- Never shuffle time-series data; the scripts handle the split correctly.
- Don't test 20 combos against the same holdout and keep the "best" — after
  many tries the winner is luck. Test few, planned combos.
- Before real money: verify the final model+threshold on a walk-forward run
  and at least a month of demo trading.
