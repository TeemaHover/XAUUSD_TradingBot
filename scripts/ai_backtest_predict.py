"""
Generate OFFLINE AI predictions for backtesting (no MT5 needed).

Slides through historical CSVs, batch-predicts with the trained model, and
writes one prediction per 5m candle to a CSV that the TypeScript backtester
consumes (npm run backtest -- data/gold_5m.csv --ai).

By default only the CHRONOLOGICAL HOLDOUT (the last --val-frac of samples,
matching ai_train.py's split) is predicted, so backtest results are
out-of-sample. Use --all to predict everything (in-sample results are
inflated — only useful for debugging).

Usage (multi-TF model):
    python scripts/ai_backtest_predict.py data/gold_5m.csv \
        --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv

Options:
    --model      Model path (default models/ai_model.npz)
    --out        Output CSV (default data/ai_predictions.csv)
    --val-frac   Holdout fraction, must match training (default 0.2)
    --forward    Forward bars used in training labels (default 5)
    --all        Predict every sample, not just the holdout
    --batch      Prediction batch size (default 512)
"""

import sys, os, csv, argparse, bisect
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_features import extract_sequence_matrix, SEQ_LEN, CONTEXT_LOOKBACK
from ai_model import load_model, MultiTFCNN, LABELS
from ai_train import load_csv, chrono_split


def main():
    ap = argparse.ArgumentParser(description="Offline AI predictions for backtesting")
    ap.add_argument("csv", help="Fast timeframe CSV (e.g. data/gold_5m.csv)")
    ap.add_argument("--csv-1h", default=None, dest="csv_1h")
    ap.add_argument("--csv-4h", default=None, dest="csv_4h")
    ap.add_argument("--model", default="models/ai_model.npz")
    ap.add_argument("--out", default="data/ai_predictions.csv")
    ap.add_argument("--val-frac", type=float, default=0.2, dest="val_frac")
    ap.add_argument("--forward", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=None,
                    help="bars dropped between train and holdout; MUST match "
                         "training (= --horizon for triple-barrier labels, "
                         "= --forward for move labels). Default: --forward")
    ap.add_argument("--all", action="store_true", help="predict all samples (in-sample!)")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model} — run ai_train.py first")

    try:
        model = load_model(args.model)
    except ValueError as e:
        sys.exit(f"Cannot use this model: {e}")

    multi_tf = isinstance(model, MultiTFCNN)
    features = model.feature_names   # extract exactly what the model was trained on
    print(f"Model: {args.model} ({'multi-TF' if multi_tf else 'single-TF'}, "
          f"{len(features)} features: {','.join(features)})")

    print(f"Loading {args.csv} ...")
    candles_5m = load_csv(args.csv)
    print(f"  {len(candles_5m):,} 5m candles")

    # --- which sample indices to predict --------------------------------
    # Sample j corresponds to candle index (SEQ_LEN + j), exactly as in
    # ai_train.py. Training trims to N = n_samples - forward, then holds out
    # the last val_frac of those N. We replicate that split here.
    n_samples = len(candles_5m) - SEQ_LEN
    if n_samples <= 0:
        sys.exit("Not enough candles")
    embargo = args.embargo if args.embargo is not None else args.forward
    N = n_samples - embargo
    _, val_idx = chrono_split(N, args.val_frac, embargo=embargo)

    pred_idx = np.arange(n_samples) if args.all else val_idx
    j0 = int(pred_idx[0])
    print(f"Predicting {len(pred_idx):,} samples "
          f"({'ALL — in-sample!' if args.all else 'holdout only'}), "
          f"candles {SEQ_LEN + j0:,}..{SEQ_LEN + int(pred_idx[-1]):,}")

    # --- 5m features (only for the needed range) -------------------------
    # Slice starts CONTEXT_LOOKBACK candles before j0 so context features
    # (S/R, OB, slopes) have full history; skip the extra leading samples.
    print("Extracting 5m features ...")
    slice_start = max(0, j0 - CONTEXT_LOOKBACK)
    X_5m = extract_sequence_matrix(candles_5m[slice_start:], window=SEQ_LEN, features=features)
    lead = j0 - slice_start
    X_5m = X_5m[lead: lead + len(pred_idx)]

    # --- multi-TF alignment ----------------------------------------------
    if multi_tf:
        csv_1h = args.csv_1h or "data/gold_1h.csv"
        csv_4h = args.csv_4h or "data/gold_4h.csv"
        print(f"Loading {csv_1h} and {csv_4h} ...")
        candles_1h = load_csv(csv_1h)
        candles_4h = load_csv(csv_4h)
        print("Extracting 1h/4h features ...")
        X_1h = extract_sequence_matrix(candles_1h, window=SEQ_LEN, features=features)
        X_4h = extract_sequence_matrix(candles_4h, window=SEQ_LEN, features=features)

        def align(candles_slow, X_slow):
            end_times = [candles_slow[j + SEQ_LEN]["time"] for j in range(len(X_slow))]
            out = np.empty(len(pred_idx), dtype=np.int64)
            for k, j in enumerate(pred_idx):
                t = candles_5m[SEQ_LEN + int(j)]["time"]
                idx = bisect.bisect_right(end_times, t) - 1
                out[k] = max(0, min(idx, len(X_slow) - 1))
            return out

        idx_1h = align(candles_1h, X_1h)
        idx_4h = align(candles_4h, X_4h)

    # --- batch predict -----------------------------------------------------
    print("Predicting ...")
    rows = []
    for start in range(0, len(pred_idx), args.batch):
        sl = slice(start, start + args.batch)
        if multi_tf:
            x = {"5m": X_5m[sl],
                 "1h": X_1h[idx_1h[sl]],
                 "4h": X_4h[idx_4h[sl]]}
        else:
            x = X_5m[sl]
        probs = model.predict(x)
        preds = np.argmax(probs, axis=1)
        for k_local, j in enumerate(pred_idx[sl]):
            i = SEQ_LEN + int(j)
            p = probs[k_local]
            d = int(preds[k_local])
            rows.append([candles_5m[i]["time"], LABELS[d], f"{p[d]:.4f}",
                         f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
        if (start // args.batch) % 20 == 0:
            print(f"  {min(start + args.batch, len(pred_idx)):,}/{len(pred_idx):,}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "direction", "confidence", "probBuy", "probSell", "probHold"])
        w.writerows(rows)

    dirs = [r[1] for r in rows]
    print(f"\nWrote {len(rows):,} predictions -> {args.out}")
    print(f"  long={dirs.count('long'):,}  short={dirs.count('short'):,}  hold={dirs.count('hold'):,}")
    print(f"\nNext: npm run build && node dist/src/backtest/runBacktest.js {args.csv} --ai")


if __name__ == "__main__":
    main()
