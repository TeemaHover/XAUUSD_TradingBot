@echo off
REM ============================================================
REM  XAUUSD bot: full data + AI + HMM pipeline, in order.
REM  Run from the project root:  run_pipeline.bat
REM
REM  Steps:
REM    1. Download 3 years of Dukascopy data (5m / 1h / 4h)
REM    2. Back up existing CSVs, convert downloads to bot format
REM    3. Train the CNN model            (SLOW - can take hours)
REM    4. Generate holdout AI predictions
REM    5. Fit HMM regimes
REM    6. Build TypeScript
REM    7. Backtest AI vs rules on the holdout
REM    8. Per-regime breakdown reports
REM ============================================================
setlocal
cd /d "%~dp0"

set FROM=2023-07-01
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TO=%%i

echo.
echo === [1/8] Downloading Dukascopy XAUUSD %FROM% to %TO% ===
call npx -y dukascopy-node -i xauusd -from %FROM% -to %TO% -t m5 -f csv -v -dir download || goto :error
call npx -y dukascopy-node -i xauusd -from %FROM% -to %TO% -t h1 -f csv -v -dir download || goto :error
call npx -y dukascopy-node -i xauusd -from %FROM% -to %TO% -t h4 -f csv -v -dir download || goto :error

echo.
echo === [2/8] Backing up old CSVs and converting ===
if not exist data mkdir data
if not exist data\backup mkdir data\backup
if exist data\gold_5m.csv copy /y data\gold_5m.csv data\backup\gold_5m.csv >nul
if exist data\gold_1h.csv copy /y data\gold_1h.csv data\backup\gold_1h.csv >nul
if exist data\gold_4h.csv copy /y data\gold_4h.csv data\backup\gold_4h.csv >nul
python scripts\convert_dukascopy.py download\xauusd-m5-*.csv data\gold_5m.csv || goto :error
python scripts\convert_dukascopy.py download\xauusd-h1-*.csv data\gold_1h.csv || goto :error
python scripts\convert_dukascopy.py download\xauusd-h4-*.csv data\gold_4h.csv || goto :error

echo.
echo === [3/8] Training CNN model (this is the slow step) ===
python scripts\ai_train.py data\gold_5m.csv --csv-1h data\gold_1h.csv --csv-4h data\gold_4h.csv || goto :error

echo.
echo === [4/8] Generating holdout AI predictions ===
python scripts\ai_backtest_predict.py data\gold_5m.csv --csv-1h data\gold_1h.csv --csv-4h data\gold_4h.csv || goto :error

echo.
echo === [5/8] Fitting HMM regimes ===
python scripts\hmm_regime.py data\gold_5m.csv || goto :error

echo.
echo === [6/8] Building TypeScript ===
call npm run build || goto :error

echo.
echo === [7/8] Backtesting AI vs rules on holdout ===
node dist\src\backtest\runBacktest.js data\gold_5m.csv --ai || goto :error

echo.
echo === [8/8] Per-regime breakdown ===
python scripts\regime_report.py backtest-ai.json data\hmm_regimes.csv || goto :error
python scripts\regime_report.py backtest-rules-holdout.json data\hmm_regimes.csv || goto :error

echo.
echo ============================================================
echo  PIPELINE COMPLETE
echo  Read in this order:
echo    1. AI vs RULES table above  - does AI beat rules? (expectancy)
echo    2. Per-regime tables        - is one state clearly losing money?
echo  Details: backtest-ai.json / backtest-rules-holdout.json
echo ============================================================
goto :eof

:error
echo.
echo ############################################################
echo  PIPELINE FAILED at the step above (exit code %errorlevel%)
echo  Fix the error and re-run. Old data is safe in data\backup\
echo ############################################################
exit /b 1
