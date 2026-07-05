"""
Per-regime backtest breakdown: joins backtest trades with HMM regimes and
shows where the strategy actually makes / loses money.

Also answers "what if I skip state k?" for every state, so you can see
whether a regime filter would improve overall expectancy.

Usage:
    python scripts/regime_report.py backtest-results.json data/hmm_regimes.csv
    python scripts/regime_report.py backtest-ai.json data/hmm_regimes.csv
"""

import sys, json, csv, bisect


def load_regimes(path):
    times, states = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            times.append(int(row["time"]))
            states.append(int(row["state"]))
    return times, states


def state_at(times, states, t):
    """Most recent regime bar at or before trade entry time t."""
    i = bisect.bisect_right(times, t) - 1
    return states[i] if i >= 0 else None


def stats(rs):
    n = len(rs)
    if n == 0:
        return {"trades": 0, "winRate": 0.0, "expectancy": 0.0,
                "pf": 0.0, "totalR": 0.0}
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gross_w = sum(wins)
    gross_l = abs(sum(losses))
    return {
        "trades": n,
        "winRate": len(wins) / n,
        "expectancy": sum(rs) / n,
        "pf": (gross_w / gross_l) if gross_l > 0 else float("inf"),
        "totalR": sum(rs),
    }


def fmt(label, s):
    pf = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "inf"
    return (f"  {label:<16} {s['trades']:>7}  {s['winRate']*100:>7.1f}%  "
            f"{pf:>7}  {s['expectancy']:>8.3f}R  {s['totalR']:>8.2f}R")


def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: python scripts/regime_report.py <backtest.json> <hmm_regimes.csv>")
    bt_path, hmm_path = sys.argv[1], sys.argv[2]

    with open(bt_path) as f:
        result = json.load(f)
    trades = result.get("trades", [])
    if not trades:
        sys.exit(f"No trades in {bt_path}")

    times, states = load_regimes(hmm_path)
    n_states = max(states) + 1

    tagged = []
    unmatched = 0
    for tr in trades:
        s = state_at(times, states, tr["entryTime"])
        if s is None:
            unmatched += 1
            continue
        tagged.append((s, float(tr["resultR"])))
    if unmatched:
        print(f"WARNING: {unmatched} trades before first regime bar — skipped")

    header = (f"  {'':<16} {'trades':>7}  {'winRate':>8}  {'PF':>7}  "
              f"{'expect':>9}  {'totalR':>9}")

    print(f"\n=== PER-REGIME BREAKDOWN ({bt_path}) ===")
    print(header)
    all_rs = [r for _, r in tagged]
    print(fmt("ALL", stats(all_rs)))
    for k in range(n_states):
        rs = [r for s, r in tagged if s == k]
        print(fmt(f"state {k}", stats(rs)))

    print(f"\n=== WHAT IF: skip one state entirely ===")
    print(header)
    base = stats(all_rs)
    for k in range(n_states):
        rs = [r for s, r in tagged if s != k]
        s = stats(rs)
        delta = s["expectancy"] - base["expectancy"]
        print(fmt(f"skip state {k}", s) + f"   (expectancy {'+' if delta >= 0 else ''}{delta:.3f}R)")

    print("\nIf skipping a state clearly raises expectancy AND that state had")
    print("meaningful trade count, add it as a filter. If deltas are tiny or")
    print("trade counts are small (<30), it's noise — don't filter on it.")


if __name__ == "__main__":
    main()
