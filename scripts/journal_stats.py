"""
Compute per-trade R statistics from the SQLite journal + MT5 deal history.

This is the scoreboard DEMO_PLAN.md asks for: average R per trade, judged
against the pre-committed thresholds (+0.05R go / -0.05R stop).

Usage (MT5 terminal must be running and logged into the account that
placed the trades):

    python scripts/journal_stats.py
    python scripts/journal_stats.py --db data/trading-journal.sqlite --days 90

Each journal trade stores entry, stop loss, volume, and the broker position
id. Realized PnL comes from MT5's deal history for that position
(profit + commission + swap over every deal), and:

    risk$ = |entry - stopLoss| * volume * contractSize
    R     = realized PnL / risk$

Trades whose position id has no closing deal yet are reported as OPEN.
Trades placed through a different account/terminal (e.g. the old MetaApi
demo) show as UNMATCHED.
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit("MetaTrader5 package required: pip install MetaTrader5")


def fmt_time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    ap = argparse.ArgumentParser(description="Average R per trade from journal + MT5 history")
    ap.add_argument("--db", default="data/trading-journal.sqlite")
    ap.add_argument("--days", type=int, default=0,
                    help="only include journal trades from the last N days (0 = all)")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    query = ("select open_time, position_id, symbol, direction, volume, entry, stop_loss "
             "from trades order by open_time")
    rows = con.execute(query).fetchall()
    con.close()

    if args.days > 0:
        cutoff = datetime.now(timezone.utc).timestamp() * 1000 - args.days * 86_400_000
        rows = [r for r in rows if r[0] >= cutoff]

    if not rows:
        sys.exit("No trades in journal for the selected window.")

    if not mt5.initialize():
        sys.exit(f"MetaTrader5 initialize failed: {mt5.last_error()} "
                 "(is the terminal running and logged in?)")

    contract_sizes: dict[str, float] = {}
    closed, open_positions, unmatched = [], [], []

    try:
        for open_time, position_id, symbol, direction, volume, entry, stop_loss in rows:
            if symbol not in contract_sizes:
                info = mt5.symbol_info(symbol)
                contract_sizes[symbol] = float(info.trade_contract_size) if info else 100.0

            risk_per_price = abs(entry - stop_loss)
            risk_usd = risk_per_price * volume * contract_sizes[symbol]

            deals = mt5.history_deals_get(position=int(position_id))
            if not deals:
                # Position id unknown to this terminal, OR it is still open.
                if mt5.positions_get(ticket=int(position_id)):
                    open_positions.append((open_time, direction, volume, entry))
                else:
                    unmatched.append((open_time, position_id, direction))
                continue

            has_exit = any(d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT,
                                       mt5.DEAL_ENTRY_OUT_BY) for d in deals)
            if not has_exit:
                open_positions.append((open_time, direction, volume, entry))
                continue

            pnl = sum(d.profit + d.commission + d.swap for d in deals)
            r = pnl / risk_usd if risk_usd > 0 else 0.0
            closed.append((open_time, direction, volume, entry, stop_loss, pnl, r))
    finally:
        mt5.shutdown()

    print(f"\nJournal trades: {len(rows)}   closed: {len(closed)}   "
          f"open: {len(open_positions)}   unmatched: {len(unmatched)}")

    if unmatched:
        print("\nUNMATCHED (no history in this terminal — different account?):")
        for open_time, position_id, direction in unmatched:
            print(f"  {fmt_time(open_time)}  #{position_id}  {direction}")

    if not closed:
        sys.exit("\nNo closed trades matched — nothing to score yet.")

    print(f"\n{'opened (UTC)':<18}{'dir':<7}{'lots':>6}{'entry':>10}{'SL':>10}"
          f"{'PnL $':>9}{'R':>8}")
    for open_time, direction, volume, entry, stop_loss, pnl, r in closed:
        print(f"{fmt_time(open_time):<18}{direction:<7}{volume:>6.2f}{entry:>10.2f}"
              f"{stop_loss:>10.2f}{pnl:>9.2f}{r:>8.2f}")

    rs = [t[6] for t in closed]
    wins = [r for r in rs if r > 0]
    avg_r = sum(rs) / len(rs)
    sorted_rs = sorted(rs)
    mid = len(sorted_rs) // 2
    median_r = sorted_rs[mid] if len(sorted_rs) % 2 else (sorted_rs[mid - 1] + sorted_rs[mid]) / 2
    total_pnl = sum(t[5] for t in closed)

    print(f"\nClosed trades : {len(closed)}")
    print(f"Win rate      : {len(wins) / len(closed):.1%}")
    print(f"Average R     : {avg_r:+.3f}")
    print(f"Median R      : {median_r:+.3f}")
    print(f"Best / worst R: {max(rs):+.2f} / {min(rs):+.2f}")
    print(f"Total PnL     : ${total_pnl:+.2f}")

    # DEMO_PLAN.md pre-committed decision rules
    print("\nDEMO_PLAN verdict (valid after 100 trades or 30 days):")
    if avg_r > 0.05:
        print(f"  avg R {avg_r:+.3f} > +0.05  ->  edge may be real; keep settings, "
              "consider small live size")
    elif avg_r < -0.05:
        print(f"  avg R {avg_r:+.3f} < -0.05  ->  STOP. Investigate execution costs "
              "first, do not add features")
    else:
        print(f"  avg R {avg_r:+.3f} in [-0.05, +0.05]  ->  inconclusive; compare "
              "fills vs backtest assumptions, extend demo")


if __name__ == "__main__":
    main()
