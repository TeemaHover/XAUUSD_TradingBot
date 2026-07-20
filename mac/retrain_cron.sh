#!/bin/bash
# Cron wrapper: monthly retrain with absolute paths (cron starts in $HOME).
# Trains a CANDIDATE model only — never auto-deploys (see mac/retrain.sh).
PROJECT="/Users/teema/Desktop/XAUUSD_TradingBot"
cd "$PROJECT" || exit 1
exec /usr/bin/caffeinate -i /bin/bash "$PROJECT/mac/retrain.sh" >> "$PROJECT/logs/retrain_monthly.log" 2>&1
