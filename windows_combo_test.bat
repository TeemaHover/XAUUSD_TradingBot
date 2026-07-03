@echo off
REM ============================================================
REM  Feature-combo experiment runner - Windows
REM  Trains each feature combination, backtests it, and collects
REM  all results into combo_results.txt for comparison.
REM
REM  Run:  windows_combo_test.bat     (leave it running - hours!)
REM  Safe to re-run: already-trained combos are skipped.
REM ============================================================
cd /d "%~dp0"

if not exist data\gold_5m.csv (
    echo No data found - run windows_auto_command.bat first.
    pause
    exit /b 1
)

echo Combo experiment started %date% %time% > combo_results.txt
echo Threshold comes from config aiConfidenceThreshold. >> combo_results.txt
echo. >> combo_results.txt

call :combo baseline "base,patterns"
call :combo all "all"
call :combo srob "base,patterns,sr_dist,ob_dist"
call :combo slopes "base,patterns,slope_high,slope_low,convergence"

call :blue "============================================================"
call :blue " ALL COMBOS DONE - results:"
call :blue "============================================================"
type combo_results.txt
echo.
echo Decision rule: keep the combo with the best EXPECTANCY that has
echo enough trades (100+). If nothing beats baseline, baseline wins.
pause
exit /b 0

REM ---------------------------------------------------------- subroutines
:combo
REM %1 = short name, %2 = feature spec
call :blue "=== COMBO %~1  (features: %~2) ==="

if exist models\m_%~1.npz (
    echo     model m_%~1.npz already exists - skipping training
) else (
    call :blue "[train] %~1 ..."
    python scripts\ai_train.py data\gold_5m.csv --csv-1h data\gold_1h.csv --csv-4h data\gold_4h.csv --label-mode triple --tp-r 2 --sl-r 1 --horizon 96 --stride 3 --features %~2 --out models\m_%~1.npz
    if errorlevel 1 (
        echo COMBO %~1 : TRAINING FAILED >> combo_results.txt
        exit /b 0
    )
)

call :blue "[predict] %~1 ..."
python scripts\ai_backtest_predict.py data\gold_5m.csv --csv-1h data\gold_1h.csv --csv-4h data\gold_4h.csv --model models\m_%~1.npz --embargo 96 --out data\preds_%~1.csv
if errorlevel 1 (
    echo COMBO %~1 : PREDICTION FAILED >> combo_results.txt
    exit /b 0
)

call :blue "[backtest] %~1 ..."
node dist\src\backtest\runBacktest.js data\gold_5m.csv --ai data\preds_%~1.csv > combo_run_%~1.txt 2>&1

echo COMBO %~1  (features: %~2) >> combo_results.txt
findstr /C:"AI (CNN predictions)" combo_run_%~1.txt >> combo_results.txt
findstr /C:"Rules (signal engine)" combo_run_%~1.txt >> combo_results.txt
echo. >> combo_results.txt
exit /b 0

:blue
echo.
powershell -NoProfile -Command "Write-Host '%~1' -ForegroundColor Blue"
exit /b 0
