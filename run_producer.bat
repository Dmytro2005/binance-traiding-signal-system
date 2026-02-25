@echo off
REM Launcher for producer

echo Starting producer
echo.

REM Setting Python path 
set PYTHONPATH=%CD%

REM Running the producer
python src\binance_producer.py

pause
