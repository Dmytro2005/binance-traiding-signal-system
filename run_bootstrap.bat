@echo off
REM Bootstrap data fetcher
REM Should be ran once before starting the system

echo Fetching the last 21 30m candles from binance and sending them to Kafka
pause

REM Set Python path
set PYTHONPATH=%CD%

REM Run bootstrap
python bootstrap_historical.py

pause
