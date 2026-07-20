# Demo Month Plan — started 2026-07-09

Written BEFORE the demo run, so decisions are made by rules, not emotions.

## Setup under test (do not change during the month)

| Setting | Value |
|---|---|
| Model | `models/ai_model.npz` (baseline: base+patterns, 12 features) |
| Confidence threshold | 0.35 (validated by sweep — model rarely exceeds 0.45) |
| Risk per trade | 5% (demo-only; $100 account, min-lot constraint — real money must use 0.5–1%) |
| Broker | MetaApi demo account |
| Phase 1 | `DRY_RUN=true` — signals logged, no orders (~1 day, verify plumbing) |
| Phase 2 | `DRY_RUN=false` — real orders on demo (rest of the month) |

## What the backtests predict (7-month holdout, same costs)

- Expectancy: **+0.13 to +0.16 R per trade** (run-to-run seed noise is ±0.03R)
- Win rate: ~54% (TP 2R / SL 1R, breakeven management)
- Max drawdown: ~10–15%
- Losing streaks of 5–6 trades are NORMAL and expected
- One losing month out of eight happened in backtest (Feb 2026) — a losing
  week proves nothing

## Abort / decision rules (pre-committed)

1. **Do not touch any setting until 100 closed trades or 30 days,**
   whichever comes first. No threshold changes, no feature changes, no
   model swaps mid-test — that would invalidate the sample.
2. After 100 trades / 30 days, compute average R per trade from the journal
   (`data/trading-journal.sqlite`):
   - **> +0.05R** → evidence the edge is real. Consider going live with the
     SAME settings and small size (0.5% risk), or extend demo.
   - **−0.05R to +0.05R** → inconclusive. Compare demo fills vs backtest
     assumptions (spread, slippage per trade). Extend demo another month.
   - **< −0.05R** → STOP. Investigate execution costs first (the usual
     culprit), not the model. Do not "fix" by adding features.
3. At 5% risk per trade, dollar drawdowns will be ~10x the backtest's
   percentages — IGNORE drawdown % this month, judge ONLY by average R per
   trade. (A normal 6-loss streak = ~-30% of this demo account. Expected.)
   Emergency stop only if the bot malfunctions (orders without stops, wrong
   sizes, repeated errors) or the account is fully blown (restart demo,
   the R-statistics in the journal survive).
4. A losing streak alone is NOT a reason to stop. 6 losses in a row has
   ~high probability of occurring at least once in 100 trades at 54% WR.

## During the month (allowed activities)

- Watch logs / Telegram alerts. Check the dashboard.
- 6-year retrain experiment (`models/m_6y.npz`) — runs in parallel, does NOT
  touch the live model. Deploy only after the demo month, only if it beats
  the baseline recipe on its own holdout.
- Keep the Mac awake (caffeinate, lid open, charger in) or move to a VPS.

## Review checklist (end of month)

- [ ] Trades closed: ____  (target ≥100)
- [ ] Average R/trade: ____  (backtest says +0.13 to +0.16)
- [ ] Win rate: ____  (backtest says ~54%)
- [ ] Max drawdown: ____  (backtest says 10–15%)
- [ ] Avg slippage+spread cost per trade vs backtest assumption
- [ ] Uptime: was the bot running during London/NY every weekday?
