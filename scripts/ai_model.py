"""
Neural network models for XAUUSD chart pattern recognition.
Pure numpy — no external ML dependencies.

CNN1D       — single-timeframe 1D CNN (30-candle sequence)
MultiTFCNN  — multi-timeframe CNN (5m + 1h + 4h parallel branches)

Architecture — MultiTFCNN:
  5m branch:  Conv1D(7→32, k=3) → Conv1D(32→64, k=5) → GlobalAvgPool → (64,)
  1h branch:  Conv1D(7→32, k=3) → Conv1D(32→64, k=5) → GlobalAvgPool → (64,)  ─► concat (192,)
  4h branch:  Conv1D(7→32, k=3) → Conv1D(32→64, k=5) → GlobalAvgPool → (64,)
              Dense(192→96) + ReLU → Dense(96→3) + Softmax

Labels:  0=BUY  1=SELL  2=HOLD
"""

import numpy as np
import os
from typing import Dict, List, Tuple

LABELS     = {0: "long", 1: "short", 2: "hold"}
SEQ_LEN    = 30
N_FEATURES = 7
N_CLASSES  = 3
MTF_TFS    = ["5m", "1h", "4h"]   # timeframe order used by MultiTFCNN
BRANCH_DIM = 64                    # GAP output size per branch


# ══════════════════════════════════════════════════════════ activations ══

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0., x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def cross_entropy(probs: np.ndarray, labels: np.ndarray,
                  class_weights: np.ndarray = None) -> float:
    n = labels.shape[0]
    log_p = np.log(probs[np.arange(n), labels] + 1e-12)
    if class_weights is not None:
        w = class_weights[labels]
        return float(-np.sum(w * log_p) / (np.sum(w) + 1e-12))
    return float(-np.mean(log_p))


# ══════════════════════════════════════════════════════════════ Conv1D ══

class Conv1DLayer:
    """
    1-D convolution. W: (kernel, in_ch, out_ch), b: (out_ch,)
    forward  (B, T, in_ch) → (B, T-K+1, out_ch)
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int, rng):
        scale = np.sqrt(2. / (kernel * in_ch))
        self.W  = rng.standard_normal((kernel, in_ch, out_ch)).astype(np.float32) * scale
        self.b  = np.zeros(out_ch, dtype=np.float32)
        self._x = None

    @property
    def kernel(self) -> int:
        return self.W.shape[0]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        B, T, _ = x.shape
        K, C_out = self.kernel, self.W.shape[2]
        out_T = T - K + 1
        patches = np.lib.stride_tricks.sliding_window_view(
            x, (1, K, x.shape[2])
        ).reshape(B, out_T, -1)
        return (patches @ self.W.reshape(-1, C_out) + self.b).astype(np.float32)

    def backward(self, dout: np.ndarray):
        x = self._x
        B, T, C_in = x.shape
        K, C_out = self.kernel, self.W.shape[2]
        out_T = T - K + 1
        patches = np.lib.stride_tricks.sliding_window_view(
            x, (1, K, C_in)
        ).reshape(B, out_T, -1)
        W_flat  = self.W.reshape(-1, C_out)
        dout_2d = dout.reshape(B * out_T, C_out)
        pat_2d  = patches.reshape(B * out_T, K * C_in)
        dW = (pat_2d.T @ dout_2d).reshape(K, C_in, C_out)
        db = dout.sum(axis=(0, 1))
        dx_patches = (dout_2d @ W_flat.T).reshape(B, out_T, K, C_in)
        dx = np.zeros_like(x)
        for t in range(out_T):
            dx[:, t:t+K, :] += dx_patches[:, t, :, :]
        return dx.astype(np.float32), dW.astype(np.float32), db.astype(np.float32)


# ══════════════════════════════════════════════════════════════ Dense ══

class DenseLayer:
    def __init__(self, in_dim: int, out_dim: int, rng):
        scale = np.sqrt(2. / in_dim)
        self.W  = rng.standard_normal((in_dim, out_dim)).astype(np.float32) * scale
        self.b  = np.zeros(out_dim, dtype=np.float32)
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return (x @ self.W + self.b).astype(np.float32)

    def backward(self, dout: np.ndarray):
        dW = self._x.T @ dout
        db = dout.sum(axis=0)
        dx = dout @ self.W.T
        return dx.astype(np.float32), dW.astype(np.float32), db.astype(np.float32)


# ══════════════════════════════════════════════════ Adam optimiser mixin ══

class AdamMixin:
    def _init_adam(self, params: List[Tuple[str, np.ndarray]]):
        self._t = 0
        self._m: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params}
        self._v: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params}

    def adam_step(self, grads: Dict[str, np.ndarray], param_map: Dict,
                  lr: float = 1e-3, beta1: float = 0.9,
                  beta2: float = 0.999, eps: float = 1e-8):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * g**2
            m_hat = self._m[k] / (1 - beta1**self._t)
            v_hat = self._v[k] / (1 - beta2**self._t)
            layer, attr = param_map[k]
            param = getattr(layer, attr)
            param -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)


# ══════════════════════════════════════════════════════════ CNN1D (v1) ══

class CNN1D(AdamMixin):
    """Single-timeframe 1D CNN. Kept for backward compatibility."""

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.conv1 = Conv1DLayer(N_FEATURES, 32, kernel=3, rng=rng)
        self.conv2 = Conv1DLayer(32, 64, kernel=5, rng=rng)
        self.fc1   = DenseLayer(64, 32, rng=rng)
        self.fc2   = DenseLayer(32, N_CLASSES, rng=rng)
        self._init_adam(self._all_params())

    def _all_params(self):
        return [
            ("conv1_W", self.conv1.W), ("conv1_b", self.conv1.b),
            ("conv2_W", self.conv2.W), ("conv2_b", self.conv2.b),
            ("fc1_W",   self.fc1.W),   ("fc1_b",   self.fc1.b),
            ("fc2_W",   self.fc2.W),   ("fc2_b",   self.fc2.b),
        ]

    def _param_map(self):
        return {
            "conv1_W": (self.conv1,"W"), "conv1_b": (self.conv1,"b"),
            "conv2_W": (self.conv2,"W"), "conv2_b": (self.conv2,"b"),
            "fc1_W":   (self.fc1,"W"),   "fc1_b":   (self.fc1,"b"),
            "fc2_W":   (self.fc2,"W"),   "fc2_b":   (self.fc2,"b"),
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.conv1.forward(x); self._z1 = h; h = relu(h)
        h = self.conv2.forward(h); self._z2 = h; h = relu(h)
        h = h.mean(axis=1); self._gap_out = h
        h = self.fc1.forward(h); self._z3 = h; h = relu(h)
        h = self.fc2.forward(h)
        self._probs = softmax(h)
        return self._probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def backward(self, x: np.ndarray, labels: np.ndarray,
                 class_weights: np.ndarray = None) -> Dict:
        B = x.shape[0]
        dz = self._probs.copy()
        dz[np.arange(B), labels] -= 1.
        if class_weights is not None:
            w = class_weights[labels].reshape(-1, 1)
            dz *= w; dz /= (w.sum() + 1e-12)
        else:
            dz /= B
        dx_fc2, dW_fc2, db_fc2 = self.fc2.backward(dz)
        dx_fc2 *= relu_grad(self._z3)
        dx_fc1, dW_fc1, db_fc1 = self.fc1.backward(dx_fc2)
        T2 = self.conv2._x.shape[1] - self.conv2.kernel + 1
        d_gap = np.broadcast_to(dx_fc1[:,np.newaxis,:], (B, T2, 64)) / T2
        d_gap = d_gap.copy().astype(np.float32) * relu_grad(self._z2)
        dx_c2, dW_c2, db_c2 = self.conv2.backward(d_gap)
        dx_c2 *= relu_grad(self._z1)
        dx_c1, dW_c1, db_c1 = self.conv1.backward(dx_c2)
        return {
            "conv1_W": dW_c1, "conv1_b": db_c1,
            "conv2_W": dW_c2, "conv2_b": db_c2,
            "fc1_W": dW_fc1,  "fc1_b": db_fc1,
            "fc2_W": dW_fc2,  "fc2_b": db_fc2,
        }

    def adam_step(self, grads, lr=1e-3, **kw):
        super().adam_step(grads, self._param_map(), lr=lr, **kw)

    @staticmethod
    def loss(probs, labels, class_weights=None):
        return cross_entropy(probs, labels, class_weights)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez(path,
                 model_type=np.array("single_tf"),
                 conv1_W=self.conv1.W, conv1_b=self.conv1.b,
                 conv2_W=self.conv2.W, conv2_b=self.conv2.b,
                 fc1_W=self.fc1.W,     fc1_b=self.fc1.b,
                 fc2_W=self.fc2.W,     fc2_b=self.fc2.b,
                 t=np.array(self._t))
        print(f"Model saved → {path}")

    def load(self, path: str):
        d = np.load(path, allow_pickle=True)
        self.conv1.W = d["conv1_W"].astype(np.float32)
        self.conv1.b = d["conv1_b"].astype(np.float32)
        self.conv2.W = d["conv2_W"].astype(np.float32)
        self.conv2.b = d["conv2_b"].astype(np.float32)
        self.fc1.W   = d["fc1_W"].astype(np.float32)
        self.fc1.b   = d["fc1_b"].astype(np.float32)
        self.fc2.W   = d["fc2_W"].astype(np.float32)
        self.fc2.b   = d["fc2_b"].astype(np.float32)
        self._t      = int(d.get("t", np.array(0)))
        self._init_adam(self._all_params())

    def predict_one(self, seq: np.ndarray) -> Tuple[str, float]:
        x = seq[np.newaxis].astype(np.float32)
        probs = self.predict(x)[0]
        idx = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx])


# ══════════════════════════════════════════════════════ MultiTFCNN (v2) ══

class MultiTFCNN(AdamMixin):
    """
    Multi-timeframe CNN — sees 5m, 1h, and 4h candles simultaneously.

    Each timeframe has its own Conv1D tower (independent weights).
    Towers are merged by concatenation before the dense classification head.

    Input:  dict  {"5m": (B,30,7), "1h": (B,30,7), "4h": (B,30,7)}
    Output: probs (B, 3)   — BUY / SELL / HOLD
    """

    TIMEFRAMES = MTF_TFS   # ["5m", "1h", "4h"]

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        # One independent conv branch per TF
        self.branches: Dict[str, Dict] = {}
        for tf in self.TIMEFRAMES:
            self.branches[tf] = {
                "conv1": Conv1DLayer(N_FEATURES, 32, kernel=3, rng=rng),
                "conv2": Conv1DLayer(32, BRANCH_DIM, kernel=5, rng=rng),
            }
        # Shared dense head
        concat_dim = BRANCH_DIM * len(self.TIMEFRAMES)   # 192
        self.fc1 = DenseLayer(concat_dim, 96, rng=rng)
        self.fc2 = DenseLayer(96, N_CLASSES, rng=rng)
        self._init_adam(self._all_params())

    # ------------------------------------------------- parameter lists
    def _all_params(self):
        params = []
        for tf in self.TIMEFRAMES:
            b = self.branches[tf]
            params += [
                (f"{tf}_conv1_W", b["conv1"].W), (f"{tf}_conv1_b", b["conv1"].b),
                (f"{tf}_conv2_W", b["conv2"].W), (f"{tf}_conv2_b", b["conv2"].b),
            ]
        params += [("fc1_W", self.fc1.W), ("fc1_b", self.fc1.b),
                   ("fc2_W", self.fc2.W), ("fc2_b", self.fc2.b)]
        return params

    def _param_map(self):
        pm = {}
        for tf in self.TIMEFRAMES:
            b = self.branches[tf]
            pm[f"{tf}_conv1_W"] = (b["conv1"], "W")
            pm[f"{tf}_conv1_b"] = (b["conv1"], "b")
            pm[f"{tf}_conv2_W"] = (b["conv2"], "W")
            pm[f"{tf}_conv2_b"] = (b["conv2"], "b")
        pm["fc1_W"] = (self.fc1, "W"); pm["fc1_b"] = (self.fc1, "b")
        pm["fc2_W"] = (self.fc2, "W"); pm["fc2_b"] = (self.fc2, "b")
        return pm

    # ------------------------------------------------- branch forward
    def _branch_fwd(self, tf: str, x: np.ndarray):
        """x: (B, T, 7) → gap: (B, 64), caches z1, z2"""
        b = self.branches[tf]
        h = b["conv1"].forward(x); z1 = h.copy(); h = relu(h)
        h = b["conv2"].forward(h); z2 = h.copy(); h = relu(h)
        gap = h.mean(axis=1)   # (B, 64)
        return gap, z1, z2

    # ------------------------------------------------- full forward pass
    def forward(self, x_dict: Dict[str, np.ndarray]) -> np.ndarray:
        gaps, self._z_cache = [], {}
        for tf in self.TIMEFRAMES:
            gap, z1, z2 = self._branch_fwd(tf, x_dict[tf])
            gaps.append(gap)
            self._z_cache[tf] = (z1, z2)
        h = np.concatenate(gaps, axis=1)   # (B, 192)
        self._concat = h
        h = self.fc1.forward(h); self._z3 = h.copy(); h = relu(h)
        h = self.fc2.forward(h)
        self._probs = softmax(h)
        return self._probs

    def predict(self, x_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return self.forward(x_dict)

    # ------------------------------------------------ backward pass
    def backward(self, x_dict: Dict[str, np.ndarray],
                 labels: np.ndarray,
                 class_weights: np.ndarray = None) -> Dict:
        B = labels.shape[0]
        grads = {}

        # Dense head gradient
        dz = self._probs.copy()
        dz[np.arange(B), labels] -= 1.
        if class_weights is not None:
            w = class_weights[labels].reshape(-1, 1)
            dz *= w; dz /= (w.sum() + 1e-12)
        else:
            dz /= B

        dx_fc2, dW_fc2, db_fc2 = self.fc2.backward(dz)
        grads["fc2_W"] = dW_fc2; grads["fc2_b"] = db_fc2

        dx_fc2 *= relu_grad(self._z3)
        dx_fc1, dW_fc1, db_fc1 = self.fc1.backward(dx_fc2)
        grads["fc1_W"] = dW_fc1; grads["fc1_b"] = db_fc1

        # Split gradient to each branch (each occupies BRANCH_DIM columns)
        for i, tf in enumerate(self.TIMEFRAMES):
            d_br = dx_fc1[:, i*BRANCH_DIM:(i+1)*BRANCH_DIM].copy()   # (B, 64)
            b    = self.branches[tf]
            z1, z2 = self._z_cache[tf]

            # un-pool (mean → broadcast)
            T2 = b["conv2"]._x.shape[1] - b["conv2"].kernel + 1
            d_gap = np.broadcast_to(d_br[:,np.newaxis,:], (B, T2, BRANCH_DIM)) / T2
            d_gap = d_gap.copy().astype(np.float32) * relu_grad(z2)

            dx_c2, dW_c2, db_c2 = b["conv2"].backward(d_gap)
            dx_c2 *= relu_grad(z1)
            dx_c1, dW_c1, db_c1 = b["conv1"].backward(dx_c2)

            grads[f"{tf}_conv1_W"] = dW_c1; grads[f"{tf}_conv1_b"] = db_c1
            grads[f"{tf}_conv2_W"] = dW_c2; grads[f"{tf}_conv2_b"] = db_c2

        return grads

    def adam_step(self, grads, lr=1e-3, **kw):
        super().adam_step(grads, self._param_map(), lr=lr, **kw)

    @staticmethod
    def loss(probs, labels, class_weights=None):
        return cross_entropy(probs, labels, class_weights)

    # -------------------------------------------- save / load
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        arrays = {"model_type": np.array("multi_tf"), "t": np.array(self._t)}
        for tf in self.TIMEFRAMES:
            b = self.branches[tf]
            arrays[f"{tf}_conv1_W"] = b["conv1"].W
            arrays[f"{tf}_conv1_b"] = b["conv1"].b
            arrays[f"{tf}_conv2_W"] = b["conv2"].W
            arrays[f"{tf}_conv2_b"] = b["conv2"].b
        arrays["fc1_W"] = self.fc1.W; arrays["fc1_b"] = self.fc1.b
        arrays["fc2_W"] = self.fc2.W; arrays["fc2_b"] = self.fc2.b
        np.savez(path, **arrays)
        print(f"Multi-TF model saved → {path}")

    def load(self, path: str):
        d = np.load(path, allow_pickle=True)
        for tf in self.TIMEFRAMES:
            b = self.branches[tf]
            b["conv1"].W = d[f"{tf}_conv1_W"].astype(np.float32)
            b["conv1"].b = d[f"{tf}_conv1_b"].astype(np.float32)
            b["conv2"].W = d[f"{tf}_conv2_W"].astype(np.float32)
            b["conv2"].b = d[f"{tf}_conv2_b"].astype(np.float32)
        self.fc1.W = d["fc1_W"].astype(np.float32)
        self.fc1.b = d["fc1_b"].astype(np.float32)
        self.fc2.W = d["fc2_W"].astype(np.float32)
        self.fc2.b = d["fc2_b"].astype(np.float32)
        self._t    = int(d.get("t", np.array(0)))
        self._init_adam(self._all_params())

    # ------------------------------------------ single-sample predict
    def predict_one(self, seqs_dict: Dict[str, np.ndarray]) -> Tuple[str, float]:
        """
        seqs_dict: {"5m": (SEQ_LEN,7), "1h": (SEQ_LEN,7), "4h": (SEQ_LEN,7)}
        Returns (direction, confidence)
        """
        x_dict = {tf: seq[np.newaxis].astype(np.float32)
                  for tf, seq in seqs_dict.items()}
        probs = self.predict(x_dict)[0]
        idx   = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx])


# ───────────────────────────── model factory ──────────────────────────────

def load_model(path: str):
    """Load either CNN1D or MultiTFCNN from .npz, detecting type automatically."""
    d = np.load(path, allow_pickle=True)
    model_type = str(d.get("model_type", np.array("single_tf")))
    if model_type == "multi_tf":
        m = MultiTFCNN()
    else:
        m = CNN1D()
    m.load(path)
    return m
