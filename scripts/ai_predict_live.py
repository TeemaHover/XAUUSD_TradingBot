"""
Broker-agnostic live AI prediction (no MetaTrader5 import).

Reads JSON from stdin:
    {
      "modelPath": "models/ai_model.npz",
      "candles": { "5m": [ {time,open,high,low,close,volume}, ... ],
                   "1h": [...], "4h": [...] }
    }

Writes JSON to stdout:
    { "direction": "long"|"short"|"hold", "confidence": 0.0-1.0, "reason"?: str }

Candles are supplied by the bot (works with MockBroker, Mt5Broker, or
MetaApiBroker). Only numpy is required:  pip3 install numpy
"""

import json
import os
import sys


def out(result):
    print(json.dumps(result))
    sys.exit(0)


def hold(reason):
    out({"direction": "hold", "confidence": 0.0, "reason": reason})


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    from ai_features import extract_sequence, extract_multi_tf_sequence, SEQ_LEN
    from ai_model import load_model, MTF_TFS
except ImportError as e:  # numpy missing, most likely
    hold(f"AI deps not available: {e} (run: pip3 install numpy)")

try:
    payload = json.loads(sys.stdin.read() or "{}")
    model_path = payload.get("modelPath", "models/ai_model.npz")
    candle_sets = payload.get("candles", {})

    if not os.path.exists(model_path):
        hold(f"model not found at {model_path} -- run ai_train.py first")

    meta = np.load(model_path, allow_pickle=True)
    model_type = str(meta.get("model_type", np.array("single_tf")))
    model = load_model(model_path)
    needed = SEQ_LEN + 1

    if model_type == "multi_tf":
        candles_dict = {}
        for tf in MTF_TFS:
            candles = candle_sets.get(tf) or []
            if len(candles) < needed:
                hold(f"need {needed} {tf} candles, got {len(candles)}")
            candles_dict[tf] = candles[-needed:]
        seqs = extract_multi_tf_sequence(candles_dict)
        direction, confidence = model.predict_one(seqs)
    else:
        candles = candle_sets.get("5m") or next(iter(candle_sets.values()), [])
        if len(candles) < needed:
            hold(f"need {needed} candles, got {len(candles)}")
        seq = extract_sequence(candles[-needed:])
        direction, confidence = model.predict_one(seq)

    out({"direction": direction, "confidence": float(confidence)})
except SystemExit:
    raise
except Exception as e:  # noqa: BLE001 — always answer with a hold
    hold(f"prediction error: {e}")
