# Crypto Trading Signal System

Real-time crypto trading signal pipeline: ingest 30-minute Binance candlesticks via WebSocket and Kafka, maintain a sliding window of 21 candles, run a stacked ML ensemble (Random Forest, GBT, Logistic Regression + meta-learner), and output **BUY** / **SELL** / **WAIT** signals with confidence. Predictions are logged to CSV and backfilled with actual outcomes when the next candle arrives.

## Overall Logic & Architecture

```
Binance WebSocket (30m closed candles)
  binance_producer.py ->  Kafka topic "binance_kline"
  bootstrap_historical.py -> same topic (last 21 candles)
  real_time_inference.py  (Spark Structured Streaming)
      Sliding window: 21 × 30m candles
      Feature generation (shared_utils)
      RF + GBT + LR → meta-learner → signal
      data/predictions.csv (+ backfill actual_label / is_correct)
```

## Installation

From the project root:

```bash
pip install kafka-python websockets requests
```

For inference you need PySpark and the Spark Kafka connector. When using `spark-submit`, add:

```bash
--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0
```

### 1. Bootstrap

Load the last 21 × 30m candles into Kafka so the inference window is ready. Without this, the first prediction would only happen after ~10.5 hours. That is how we get our first signal 

**Windows (batch):**
```bat
run_bootstrap.bat
```

**Or manually:**
```bash
set PYTHONPATH=%CD%
python bootstrap_historical.py
```

Options: `--no-kafka` (only save JSON), `--no-json` (only send to Kafka).

### 2. Start the producer

Stream live 30m closed candles from Binance to Kafka.

**Windows:**
```bat
run_producer.bat
```

**Or:**
```bash
set PYTHONPATH=%CD%
python src/binance_producer.py
```

Leave this running. New candles are published every 30 minutes.

### 3. Start real-time inference

Consumes from Kafka, fills the 21-candle window, runs the models, and prints signals.

**Windows:**  
Set the Kafka connector package to match your Spark, then:

```bat
run_inference.bat
```

**Or with spark-submit:**
```bash
set PYTHONPATH=%CD%
set PYSPARK_PYTHON=python
set PYSPARK_DRIVER_PYTHON=python
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 src/real_time_inference.py
```

Inference reads from the earliest offset so it processes the bootstrap candles first, then continues with live data. The first prediction appears after the window is full (21 candles). Each new 30m candle triggers a new prediction and backfills the previous row in `data/predictions.csv`.

## Project Structure

| `binance_kline_chk/` | Spark streaming checkpoint (do not delete if you need recovery) |

| `data/bootstrap/` | Bootstrap candle JSON (e.g. `BTCUSDT_30m_bootstrap.json`) |
| `data/predictions.csv` | Logged predictions + backfilled `actual_label`, `is_correct` |
| `data/raw.csv` | Raw data from Kaggle "Binance Full History" |


| `src/binance_producer.py` | WebSocket → Kafka producer (Binance 30m klines) |
| `src/real_time_inference.py` | Spark streaming consumer, sliding window, ML inference, signals |
| `src/shared_utils.py` | Candle aggregation, feature generation, probability UDF |
| `src/bootstrap_historical.py` | Fetch last 21 × 30m candles from Binance REST and send to Kafka |

| `docs/` | Documentation (e.g. `binance_producer.ipynb`, `real_time_inference.ipynb`, `shared_utils.ipynb`, `bootstrap_historical_notebook.ipynb`) |

| `models/` | Trained Spark ML models (stacked_rf, stacked_gbt, stacked_lr, stacked_meta) |

| `src/semi-final.ipynb` | Training notebook (stacked ensemble) |

| `run_*.bat` | Windows launchers for bootstrap, producer, inference |


## Output

- **Console:** Trading signal (BUY/SELL/WAIT), probability, confidence (VERY HIGH / HIGH / Low).
- **`data/predictions.csv`:** Columns: `timestamp`, `datetime`, `symbol`, `close`, `final_prob`, `prediction`, `actual_label`, `is_correct`. The last two are filled when the next candle arrives.

## Documentation

- `docs/binance_producer.ipynb` — Section-by-section walkthrough of the producer.
- `docs/real_time_inference.ipynb` — Walkthrough of the inference pipeline and backfill.
- `docs/shared_utils.ipynb` — Aggregation and feature generation.

Training is done in `semi-final.ipynb`; the saved models in `models/` are produced there.
