"""
Train XAUUSD multi-timeframe or single-timeframe CNN model.

Single-TF (backward compatible):
    python scripts/ai_train.py data/gold_5m.csv

Multi-TF (sees 5m + 1h + 4h simultaneously):
    python scripts/ai_train.py data/gold_5m.csv \
        --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv

Options:
    --epochs      Training epochs (default 300)
    --lr          Learning rate (default 0.001)
    --forward     Forward bars for label generation (default 5)
    --threshold   ATR multiplier for BUY/SELL threshold (default 1.0)
    --out         Output model path (default models/ai_model.npz)
"""

import sys, os, csv, argparse, bisect
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ai_features import extract_sequence_matrix, align_tf_index, SEQ_LEN
from ai_model    import CNN1D, MultiTFCNN, cross_entropy


# ---------------------------------------------------------------- data load
def load_csv(path: str):
    candles = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            candles.append({
                "time":   int(float(row["time"])),
                "open":   float(row["open"]),
                "high":   float(row["high"]),
                "low":    float(row["low"]),
                "close":  float(row["close"]),
                "volume": float(row.get("volume", 1)),
            })
    return candles


# -------------------------------------------------------------- labelling
def make_labels(candles, feature_start: int, forward: int,
                threshold_atr: float) -> np.ndarray:
    closes = [c["close"] for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    labels = []
    for i in range(feature_start, len(candles)):
        subset = candles[max(0, i-14):i+1]
        trs = [max(subset[j]["high"] - subset[j]["low"],
                   abs(subset[j]["high"] - subset[j-1]["close"]),
                   abs(subset[j]["low"]  - subset[j-1]["close"]))
               for j in range(1, len(subset))]
        atr_val   = float(np.mean(trs)) if trs else 1.0
        threshold = threshold_atr * atr_val
        end = min(i + forward + 1, len(candles))
        if end <= i + 1:
            labels.append(2); continue
        fut_high  = max(highs[i+1:end])
        fut_low   = min(lows[i+1:end])
        up_move   = fut_high - closes[i]
        down_move = closes[i] - fut_low
        if   up_move >= threshold and up_move > down_move:   labels.append(0)
        elif down_move >= threshold and down_move > up_move: labels.append(1)
        else:                                                 labels.append(2)
    return np.array(labels, dtype=np.int32)


# ---------------------------------------------------------- class weights
def compute_class_weights(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    counts  = np.bincount(y, minlength=n_classes).astype(float)
    counts  = np.where(counts == 0, 1., counts)
    weights = y.shape[0] / (n_classes * counts)
    return weights.astype(np.float32)


# --------------------------------------------------------- confusion matrix
def print_confusion(y_true, y_pred, labels=("BUY","SELL","HOLD")):
    n  = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    col_w  = 8
    header = "".join(f"{l:>{col_w}}" for l in labels)
    print(f"\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"{'':>12}{header}")
    for i, l in enumerate(labels):
        row = "".join(f"{cm[i,j]:>{col_w}}" for j in range(n))
        print(f"  {l:>8}  {row}")
    print("\nPer-class metrics:")
    for i, l in enumerate(labels):
        tp   = cm[i, i]
        fp   = cm[:, i].sum() - tp
        fn   = cm[i, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        print(f"  {l:>6}:  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}"
              f"  (actual={cm[i].sum():,}  predicted={cm[:,i].sum():,})")


# ================================================ single-TF training loop
def train_single(X: np.ndarray, y: np.ndarray, epochs: int, lr: float,
                 class_weights: np.ndarray, batch_size: int = 32) -> CNN1D:
    model      = CNN1D()
    n          = X.shape[0]
    best_loss  = float("inf")
    no_improve = 0
    current_lr = lr

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n)
        total_loss, batches = 0., 0
        for start in range(0, n, batch_size):
            bi    = idx[start:start+batch_size]
            Xb, yb = X[bi], y[bi]
            probs = model.forward(Xb)
            loss  = model.loss(probs, yb, class_weights)
            grads = model.backward(Xb, yb, class_weights)
            model.adam_step(grads, lr=current_lr)
            total_loss += loss; batches += 1

        epoch_loss = total_loss / batches
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss; no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 30:
            current_lr *= 0.5; no_improve = 0
            print(f"  [LR decay] → {current_lr:.6f}")

        if epoch % 20 == 0 or epoch == 1:
            preds  = np.argmax(model.predict(X), axis=1)
            acc    = float(np.mean(preds == y))
            counts = np.bincount(preds, minlength=3)
            print(f"  Epoch {epoch:>4}/{epochs}  loss={epoch_loss:.4f}"
                  f"  acc={acc:.3f}  BUY={counts[0]:,} SELL={counts[1]:,} HOLD={counts[2]:,}"
                  f"  lr={current_lr:.5f}")

    return model


# ============================================= multi-TF training loop
def train_multi(X_dict: dict, y: np.ndarray, epochs: int, lr: float,
                class_weights: np.ndarray, batch_size: int = 32) -> MultiTFCNN:
    model      = MultiTFCNN()
    n          = y.shape[0]
    best_loss  = float("inf")
    no_improve = 0
    current_lr = lr
    tfs        = list(X_dict.keys())

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n)
        total_loss, batches = 0., 0
        for start in range(0, n, batch_size):
            bi  = idx[start:start+batch_size]
            yb  = y[bi]
            Xb  = {tf: X_dict[tf][bi] for tf in tfs}
            probs = model.forward(Xb)
            loss  = model.loss(probs, yb, class_weights)
            grads = model.backward(Xb, yb, class_weights)
            model.adam_step(grads, lr=current_lr)
            total_loss += loss; batches += 1

        epoch_loss = total_loss / batches
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss; no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 30:
            current_lr *= 0.5; no_improve = 0
            print(f"  [LR decay] → {current_lr:.6f}")

        if epoch % 20 == 0 or epoch == 1:
            preds  = np.argmax(model.predict(X_dict), axis=1)
            acc    = float(np.mean(preds == y))
            counts = np.bincount(preds, minlength=3)
            print(f"  Epoch {epoch:>4}/{epochs}  loss={epoch_loss:.4f}"
                  f"  acc={acc:.3f}  BUY={counts[0]:,} SELL={counts[1]:,} HOLD={counts[2]:,}"
                  f"  lr={current_lr:.5f}")

    return model


# -------------------------------------------------------------------  main
def main():
    ap = argparse.ArgumentParser(description="Train XAUUSD CNN trading model")
    ap.add_argument("csv",         help="Fast timeframe CSV (e.g. data/gold_5m.csv)")
    ap.add_argument("--csv-1h",    default=None, dest="csv_1h",
                    help="1h CSV for multi-TF training (e.g. data/gold_1h.csv)")
    ap.add_argument("--csv-4h",    default=None, dest="csv_4h",
                    help="4h CSV for multi-TF training (e.g. data/gold_4h.csv)")
    ap.add_argument("--epochs",    type=int,   default=300)
    ap.add_argument("--lr",        type=float, default=0.001)
    ap.add_argument("--forward",   type=int,   default=5,
                    help="Forward bars for label generation (default 5)")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="ATR multiplier for BUY/SELL threshold (default 1.0)")
    ap.add_argument("--out",       default="models/ai_model.npz")
    args = ap.parse_args()

    multi_tf = (args.csv_1h is not None) or (args.csv_4h is not None)

    print(f"Loading {args.csv} ...")
    candles_5m = load_csv(args.csv)
    print(f"  {len(candles_5m):,} 5m candles")

    print(f"Extracting 5m sequences (window={SEQ_LEN}) ...")
    X_5m = extract_sequence_matrix(candles_5m, window=SEQ_LEN)
    print(f"  5m matrix: {X_5m.shape}")

    print(f"Generating labels (forward={args.forward}, threshold={args.threshold}x ATR) ...")
    y = make_labels(candles_5m, feature_start=SEQ_LEN,
                    forward=args.forward, threshold_atr=args.threshold)
    # Trim labels to valid training range (exclude last `forward` samples)
    N = min(len(X_5m), len(y) - args.forward)
    X_5m = X_5m[:N]; y = y[:N]
    counts = np.bincount(y, minlength=3)
    pct    = counts / counts.sum() * 100
    print(f"  BUY={counts[0]:,} ({pct[0]:.1f}%)  "
          f"SELL={counts[1]:,} ({pct[1]:.1f}%)  "
          f"HOLD={counts[2]:,} ({pct[2]:.1f}%)")

    cw = compute_class_weights(y)
    print(f"  Class weights — BUY={cw[0]:.2f}  SELL={cw[1]:.2f}  HOLD={cw[2]:.2f}")

    if multi_tf:
        # ── MULTI-TIMEFRAME ──────────────────────────────────────────────
        csv_1h = args.csv_1h or "data/gold_1h.csv"
        csv_4h = args.csv_4h or "data/gold_4h.csv"

        print(f"\nLoading slow timeframes ...")
        candles_1h = load_csv(csv_1h)
        candles_4h = load_csv(csv_4h)
        print(f"  1h: {len(candles_1h):,} candles   4h: {len(candles_4h):,} candles")

        print("Extracting 1h and 4h sequences ...")
        X_1h = extract_sequence_matrix(candles_1h, window=SEQ_LEN)
        X_4h = extract_sequence_matrix(candles_4h, window=SEQ_LEN)
        print(f"  1h matrix: {X_1h.shape}   4h matrix: {X_4h.shape}")

        # Precompute window-end timestamps for alignment
        times_1h_end = [candles_1h[j + SEQ_LEN]["time"] for j in range(len(X_1h))]
        times_4h_end = [candles_4h[j + SEQ_LEN]["time"] for j in range(len(X_4h))]

        print("Aligning timeframes by timestamp ...")
        idx_1h = align_tf_index(candles_5m, X_1h, times_1h_end)
        idx_4h = align_tf_index(candles_5m, X_4h, times_4h_end)

        # Trim to aligned length
        N = min(N, len(idx_1h), len(idx_4h))
        X_5m   = X_5m[:N]
        y      = y[:N]
        idx_1h = idx_1h[:N]
        idx_4h = idx_4h[:N]

        X_dict = {
            "5m": X_5m,
            "1h": X_1h[idx_1h],
            "4h": X_4h[idx_4h],
        }

        print(f"\nTraining MultiTFCNN ({N:,} samples, {args.epochs} epochs) ...")
        print("  Architecture: 5m+1h+4h branches → concat → dense head\n")
        model = train_multi(X_dict, y, epochs=args.epochs,
                            lr=args.lr, class_weights=cw)

        print("\nFinal evaluation:")
        preds = np.argmax(model.predict(X_dict), axis=1)
        acc   = float(np.mean(preds == y))
        print(f"  Overall accuracy: {acc:.3f}")
        print_confusion(y, preds)
        model.save(args.out)

    else:
        # ── SINGLE-TIMEFRAME (backward compat) ───────────────────────────
        print(f"\nTraining CNN1D ({N:,} samples, {args.epochs} epochs, single-TF) ...")
        model = train_single(X_5m, y, epochs=args.epochs,
                             lr=args.lr, class_weights=cw)
        print("\nFinal evaluation:")
        preds = np.argmax(model.predict(X_5m), axis=1)
        acc   = float(np.mean(preds == y))
        print(f"  Overall accuracy: {acc:.3f}")
        print_confusion(y, preds)
        model.save(args.out)

    print(f"\nDone. Run: npx ts-node src/main.ts --mode ai")

if __name__ == "__main__":
    main()
