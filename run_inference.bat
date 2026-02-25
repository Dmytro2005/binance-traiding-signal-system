@echo off
echo Starting real-time inference
echo.

REM Set Python path
set PYTHONPATH=%CD%
set PYSPARK_PYTHON=python
set PYSPARK_DRIVER_PYTHON=python
set PYSPARK_SUBMIT_ARGS=--packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 pyspark-shell
python src\real_time_inference.py
REM Run with Kafka connector
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:3.2.0 src\real_time_inference.py

pause
