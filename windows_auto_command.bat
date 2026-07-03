@echo off
REM ============================================================
REM  Full AI training pipeline - Windows
REM  Double-click this file or run:  windows_auto_command.bat
REM  Steps: deps -> data -> build -> train -> predict -> backtest -> sweep
REM ============================================================
cd /d "%~dp0"

call :step "[1/7] Checking dependencies..."
python --version || goto :fail
node --version || goto :fail
pip install numpy --quiet || goto :fail

call :step "[2/7] Downloading data (skipped if already present)..."
if exist data\gold_5m.csv (
    echo     data\gold_5m.csv already exists - skipping download
    goto :data_ok
)
echo     Trying Dukascopy first...
python scripts\download_data.py 3
if not errorlevel 1 goto :data_ok
echo.
echo     Dukascopy failed - falling back to MetaTrader5 export...
echo     (MT5 terminal must be running and logged in)
pip install MetaTrader5 --quiet
python scripts\export_mt5_data.py || goto :fail
if not exist data\gold_5m.csv goto :fail
:data_ok

call :step "[3/7] Building TypeScript backtester..."
call npm install --silent || goto :fail
call npm run build || goto :fail

call :step "[4/7] Training baseline model (this takes a LONG time - hours on CPU)..."
python scripts\ai_train.py data\gold_5m.csv --csv-1h data\gold_1h.csv --csv-4h data\gold_4h.csv --label-mode triple --tp-r 2 --sl-r 1 --horizon 96 --stride 3 --features base,patterns --out models\m_baseline.npz || goto :fail

call :step "[5/7] Generating holdout predictions..."
python scripts\ai_backtest_predict.py data\gold_5m.csv --csv-1h data\gold_1h.csv --csv-4h data\gold_4h.csv --model models\m_baseline.npz --embargo 96 --out data\preds_baseline.csv || goto :fail

call :step "[6/7] Backtesting AI vs rules..."
node dist\src\backtest\runBacktest.js data\gold_5m.csv --ai data\preds_baseline.csv || goto :fail

call :step "[7/7] Confidence threshold sweep..."
node dist\src\backtest\runBacktest.js data\gold_5m.csv --ai data\preds_baseline.csv --sweep || goto :fail

call :step "============================================================"
call :step " DONE. Compare expectancy in the tables above."
call :step " Details: backtest-ai.json / backtest-rules-holdout.json"
call :step "============================================================"
pause
exit /b 0

:step
REM prints the step header in dodger blue (nearest console blue)
echo.
powershell -NoProfile -Command "Write-Host '%~1' -ForegroundColor Blue"
exit /b 0

:fail
echo.
echo ============================================================
echo  FAILED - see the error above. Fix it and run again.
echo  (Common: no internet, Node.js not installed, or missing data)
echo ============================================================
pause
exit /b 1
