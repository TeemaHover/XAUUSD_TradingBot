"""
Collect historical candle data from MT5 for AI training.

Usage:
    # Single timeframe
    python scripts/ai_collect.py --timeframe 5m --years 3

    # All timeframes at once (required for multi-TF training)
    python scripts/ai_collect.py --multi --years 3

    # Custom symbol
    python scripts/ai_collect.py --multi --symbol GOLD --years 3

Output files (--multi):
    data/gold_5m.csv   — entry timeframe
    data/gold_1h.csv   — trend context
    data/gold_4h.csv   — higher timeframe structure

Then train:
    python scripts/ai_train.py data/gold_5m.csv --csv-1h data/gold_1h.csv --csv-4h data/gold_4h.csv

Bar counts by timeframe (approximate):
    5m:  1 year ≈  72 000 bars    3 years ≈ 216 000 bars
    1h:  1 year ≈   6 000 bars    3 years ≈  18 000 bars
    4h:  1 year ≈   1 500 bars    3 years ≈   4 500 bars
"""

import json, os, sys, csv, subprocess, argparse
from datetime import datetime, timezone, timedelta

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "default.json")
BRIDGE_PATH = os.path.join(os.path.dirname(__file__), "mt5_bridge.py")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def call_bridge(python_path: str, command: str, args: dict) -> dict:
    payload = json.dumps({"command": command, "args": args})
    result  = subprocess.run(
        [python_path, BRIDGE_PATH],
        input=payload, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        detail = ""
        if result.stdout.strip():
            try:
                detail = json.loads(result.stdout).get("error", result.stdout.strip())
            except Exception:
                detail = result.stdout.strip()
        if result.stderr.strip():
            detail += "\n" + result.stderr.strip()
        raise RuntimeError(f"Bridge error: {detail}")
    data = json.loads(result.stdout)
    if not data.get("ok"):
        raise RuntimeError(f"Bridge error: {data.get('error')}")
    return data["result"]


def fetch_and_save(python_path: str, symbol: str, tf: str,
                   years: float, bars: int, out_path: str):
    bridge_args: dict = {"symbol": symbol, "timeframe": tf}

    if bars is not None:
        bridge_args["limit"] = bars
        desc = f"{bars} bars"
    else:
        from_dt = datetime.now(timezone.utc) - timedelta(days=365.25 * years)
        bridge_args["fromTimestamp"] = int(from_dt.timestamp() * 1000)
        desc = f"{years} year(s) from {from_dt.strftime('%Y-%m-%d')}"

    print(f"  [{tf}] Fetching {symbol} [{desc}] ...", flush=True)

    result  = call_bridge(python_path, "candles", bridge_args)
    candles = result["candles"]
    if not candles:
        raise RuntimeError(f"No candles returned for {symbol} {tf}")

    candles.sort(key=lambda c: c["time"])
    first = datetime.fromtimestamp(candles[0]["time"]  / 1000, tz=timezone.utc)
    last  = datetime.fromtimestamp(candles[-1]["time"] / 1000, tz=timezone.utc)
    print(f"  [{tf}] {len(candles):,} candles  "
          f"({first.strftime('%Y-%m-%d')} → {last.strftime('%Y-%m-%d')})")

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c["time"], c["open"], c["high"], c["low"], c["close"], c["volume"]])
    print(f"  [{tf}] Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Collect MT5 candle data for AI training")
    ap.add_argument("--symbol",    default=None,  help="Symbol (default from config)")
    ap.add_argument("--timeframe", default="5m",  help="Single-TF mode: timeframe (default 5m)")
    ap.add_argument("--multi",     action="store_true",
                    help="Collect all 3 timeframes (5m, 1h, 4h) for multi-TF training")
    ap.add_argument("--years",     type=float, default=3.0,
                    help="Years of history (default 3.0)")
    ap.add_argument("--bars",      type=int,   default=None,
                    help="Exact bar count override (overrides --years)")
    ap.add_argument("--out",       default=None,
                    help="Output CSV path (single-TF mode only)")
    args = ap.parse_args()

    config      = load_config()
    symbol      = args.symbol or config.get("symbol", "GOLD")
    python_path = config.get("mt5", {}).get("pythonPath", "python")

    if args.multi:
        print(f"Collecting multi-TF data for {symbol}  ({args.years} year(s)) ...")
        print("  MT5 must be open and logged in.\n")
        tfs = ["5m", "1h", "4h"]
        errors = []
        for tf in tfs:
            out_path = os.path.join("data", f"{symbol.lower()}_{tf}.csv")
            try:
                fetch_and_save(python_path, symbol, tf, args.years, args.bars, out_path)
            except Exception as e:
                print(f"  [{tf}] ERROR: {e}")
                errors.append(tf)
        print()
        if errors:
            print(f"Failed timeframes: {errors}")
            print("Make sure MT5 is open and the symbol is in Market Watch.")
            sys.exit(1)
        sym = symbol.lower()
        print("Done! To train the multi-TF model:")
        print(f"  python scripts/ai_train.py data/{sym}_5m.csv "
              f"--csv-1h data/{sym}_1h.csv --csv-4h data/{sym}_4h.csv")
    else:
        # Single-TF mode (original behaviour)
        tf       = args.timeframe
        out_path = args.out or os.path.join("data", f"{symbol.lower()}_{tf}.csv")
        print(f"Fetching {tf} candles for {symbol} ...")
        try:
            fetch_and_save(python_path, symbol, tf, args.years, args.bars, out_path)
        except Exception as e:
            print(f"\nERROR: {e}")
            print("Make sure MT5 is open and the symbol is in Market Watch.")
            sys.exit(1)
        print(f"\nNext step:")
        print(f"  python scripts/ai_train.py {out_path}")

if __name__ == "__main__":
    main()
