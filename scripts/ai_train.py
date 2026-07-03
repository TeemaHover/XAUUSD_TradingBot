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
    --val-frac    Fraction of data held out for validation (default 0.2)
    --out         Output model path (default models/ai_model.npz)

Validation is CHRONOLOGICAL: the last `val-frac` of the data is held out,
never shuffled into training, with an embargo gap of `forward` bars so
train labels cannot peek into the validation window. Early stopping and
the saved model are based on validation loss, and the reported metrics
are out-of-sample.
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


# ------------------------------------------------- chronological split
def chrono_split(n: int, val_frac: float, embargo: int):
    """
    Time-ordered split. First (1 - val_frac) of samples -> train,
    last val_frac -> validation. `embargo` samples are dropped between
    the two so that forward-looking train labels cannot overlap the
    validation window (label leakage).
    """
    n_val   = max(1, int(round(n * val_frac)))
    v_start = n - n_val
    t_end   = max(0, v_start - embargo)
    return np.arange(0, t_end), np.arange(v_start, n)


# ---------------------------------------------------------- class weights
def compute_class_weights(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    counts  = np.bincount(y, minlength=n_classes).astype(float)
    counts  = np.where(counts == 0, 1., counts)
    weights = y.shape[0] / (n_classes * counts)
    return weights.astype(np.float32)


# ------------------------------------------------- weight snapshot/restore
def snapshot_params(model) -> dict:
    return {k: getattr(layer, attr).copy()
            for k, (layer, attr) in model._param_map().items()}


def restore_params(model, snap: dict):
    for k, (layer, attr) in model._param_map().items():
        setattr(layer, attr, snap[k].copy())


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


# =========================================================== training loop
def train_model(model, get_batch, predict_val, n_train: int, y_val: np.ndarray,
                epochs: int, lr: float, class_weights: np.ndarray,
                batch_size: int = 32,
                lr_patience: int = 15, stop_patience: int = 40):
    """
    Generic loop used by both single-TF and multi-TF models.

    get_batch(indices)  -> (Xb, yb) training batch
    predict_val()       -> probs over the validation set

    Early stopping and LR decay are driven by VALIDATION loss, and the
    best-validation weights are restored before returning.
    """
    best_val   = float("inf")
    best_snap  = snapshot_params(model)
    best_epoch = 0
    no_improve = 0
    current_lr = lr

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n_train)
        total_loss, batches = 0., 0
        for start in range(0, n_train, batch_size):
            bi     = idx[start:start+batch_size]
            Xb, yb = get_batch(bi)
            probs  = model.forward(Xb)
            loss   = model.loss(probs, yb, class_weights)
            grads  = model.backward(Xb, yb, class_weights)
            model.adam_step(grads, lr=current_lr)
            total_loss += loss; batches += 1

        train_loss = total_loss / batches
        val_probs  = predict_val()
        val_loss   = cross_entropy(val_probs, y_val)   # unweighted: honest OOS loss

        # Real explosion detector (spurious Accelerate warnings are silenced,
        # so verify the actual numbers).
        if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
            print(f"  [ABORT] non-finite loss at epoch {epoch} "
                  f"(train={train_loss}, val={val_loss}) — training diverged. "
                  f"Restoring best weights from epoch {best_epoch}.")
            break

        if val_loss < best_val - 1e-4:
            best_val   = val_loss
            best_snap  = snapshot_params(model)
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve % lr_patience == 0:
                current_lr *= 0.5
                print(f"  [LR decay] -> {current_lr:.6f}")
            if no_improve >= stop_patience:
                print(f"  [Early stop] epoch {epoch}: no val improvement "
                      f"for {stop_patience} epochs (best epoch {best_epoch})")
                break

        if epoch % 20 == 0 or epoch == 1:
            val_preds = np.argmax(val_probs, axis=1)
            val_acc   = float(np.mean(val_preds == y_val))
            counts    = np.bincount(val_preds, minlength=3)
            print(f"  Epoch {epoch:>4}/{epochs}  train_loss={train_loss:.4f}"
                  f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
                  f"  val BUY={counts[0]:,} SELL={counts[1]:,} HOLD={counts[2]:,}"
                  f"  lr={current_lr:.5f}")

    restore_params(model, best_snap)
    print(f"\n  Restored best weights (epoch {best_epoch}, val_loss={best_val:.4f})")
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
    ap.add_argument("--val-frac",  type=float, default=0.2, dest="val_frac",
                    help="Chronological validation fraction (default 0.2)")
    ap.add_argument("--out",       default="models/ai_model.npz")
    args = ap.parse_args()

    if not (0.05 <= args.val_frac <= 0.5):
        sys.exit("--val-frac must be between 0.05 and 0.5")

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

    if multi_tf:
        # -- MULTI-TIMEFRAME --------------------------------------------
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
        tfs = list(X_dict.keys())

        # Chronological split with embargo = forward bars (label lookahead)
        tr_idx, va_idx = chrono_split(N, args.val_frac, embargo=args.forward)
        X_tr = {tf: X_dict[tf][tr_idx] for tf in tfs}
        X_va = {tf: X_dict[tf][va_idx] for tf in tfs}
        y_tr, y_va = y[tr_idx], y[va_idx]
        print(f"\nChronological split: train={len(tr_idx):,}  "
              f"embargo={args.forward}  val={len(va_idx):,} (last {args.val_frac:.0%})")

        cw = compute_class_weights(y_tr)   # weights from TRAIN only
        print(f"  Class weights (train) - BUY={cw[0]:.2f}  SELL={cw[1]:.2f}  HOLD={cw[2]:.2f}")

        print(f"\nTraining MultiTFCNN ({len(tr_idx):,} samples, up to {args.epochs} epochs) ...")
        print("  Architecture: 5m+1h+4h branches -> concat -> dense head\n")
        model = MultiTFCNN()
        train_model(
            model,
            get_batch   = lambda bi: ({tf: X_tr[tf][bi] for tf in tfs}, y_tr[bi]),
            predict_val = lambda: model.predict(X_va),
            n_train     = len(tr_idx),
            y_val       = y_va,
            epochs      = args.epochs,
            lr          = args.lr,
            class_weights = cw,
        )

        print("\nOut-of-sample evaluation (validation set - never trained on):")
        val_preds = np.argmax(model.predict(X_va), axis=1)
        val_acc   = float(np.mean(val_preds == y_va))
        tr_preds  = np.argmax(model.predict(X_tr), axis=1)
        tr_acc    = float(np.mean(tr_preds == y_tr))
        print(f"  Validation accuracy: {val_acc:.3f}   (train accuracy: {tr_acc:.3f} - "
              f"large gap = overfitting)")
        print_confusion(y_va, val_preds)
        model.save(args.out)

    else:
        # -- SINGLE-TIMEFRAME (backward compat) --------------------------
        tr_idx, va_idx = chrono_split(N, args.val_frac, embargo=args.forward)
        X_tr, y_tr = X_5m[tr_idx], y[tr_idx]
        X_va, y_va = X_5m[va_idx], y[va_idx]
        print(f"\nChronological split: train={len(tr_idx):,}  "
              f"embargo={args.forward}  val={len(va_idx):,} (last {args.val_frac:.0%})")

        cw = compute_class_weights(y_tr)   # weights from TRAIN only
        print(f"  Class weights (train) - BUY={cw[0]:.2f}  SELL={cw[1]:.2f}  HOLD={cw[2]:.2f}")

        print(f"\nTraining CNN1D ({len(tr_idx):,} samples, up to {args.epochs} epochs, single-TF) ...")
        model = CNN1D()
        train_model(
            model,
            get_batch   = lambda bi: (X_tr[bi], y_tr[bi]),
            predict_val = lambda: model.predict(X_va),
            n_train     = len(tr_idx),
            y_val       = y_va,
            epochs      = args.epochs,
            lr          = args.lr,
            class_weights = cw,
        )

        print("\nOut-of-sample evaluation (validation set - never trained on):")
        val_preds = np.argmax(model.predict(X_va), axis=1)
        val_acc   = float(np.mean(val_preds == y_va))
        tr_preds  = np.argmax(model.predict(X_tr), axis=1)
        tr_acc    = float(np.mean(tr_preds == y_tr))
        print(f"  Validation accuracy: {val_acc:.3f}   (train accuracy: {tr_acc:.3f} - "
              f"large gap = overfitting)")
        print_confusion(y_va, val_preds)
        model.save(args.out)

    print(f"\nDone. Run: npx ts-node src/main.ts --mode ai")

if __name__ == "__main__":
    main()
