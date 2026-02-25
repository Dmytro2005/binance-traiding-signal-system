#consumes 1m candles from kafka, aggregates them into 30m candles generates trading signals using the trained model
import os
from collections import deque
from typing import Dict, List, Optional
import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import (
    RandomForestClassificationModel,
    GBTClassificationModel,
    LogisticRegressionModel)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType

#importing shared utils depending on where they end up when the project is done
try:
    from shared_utils import generate_features, aggregate_candles, extract_prob_udf
except ImportError:
    from src.shared_utils import generate_features, aggregate_candles, extract_prob_udf

#log config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


CANDLE_MINUTES = 30  
LOOKBACK = 20  
WINDOW_SIZE = LOOKBACK + 1 


class CandlesProcessor:
    #buffers 30 1m candles, aggregates into 1 30m candle, maintains sliding window of 21 30m candles, generates features when window is full, runs prediction, outputs trading signal, backfills label when next candle arrives
    def __init__(
        self,
        model_dir: str = "models",
        predictions_dir: str = "data/live_predictions_parquet"
    ):
        #init real-time processor
        self.spark = self._initialize_spark()
        
        #buffers
        self.minute_buffer: List[Dict] = []  #1m buffer
        self.window_30m = deque(maxlen=WINDOW_SIZE) #30m window
        
        #prediction tracking
        self.predictions_path = predictions_dir
        self.pending_prediction: Optional[Dict] = None
        
        #loading trained models
        logger.info("Loading trained models")
        self.rf_model = RandomForestClassificationModel.load(
            os.path.join(model_dir, "stacked_rf_model")
        )
        self.gbt_model = GBTClassificationModel.load(
            os.path.join(model_dir, "stacked_gbt_model")
        )
        self.lr_model = LogisticRegressionModel.load(
            os.path.join(model_dir, "stacked_lr_model")
        )
        self.meta_model = LogisticRegressionModel.load(
            os.path.join(model_dir, "stacked_meta_model")
        )
        logger.info("All models loaded successfully")
        
        #creating prediction dir
        os.makedirs(predictions_dir, exist_ok=True)
    
    def _initialize_spark(self) -> SparkSession:
        return (SparkSession.builder
            .appName("CryptoTradingSignals_RealTime")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "2g")
            .config("spark.executor.memory", "2g")
            .config("spark.sql.codegen.wholeStage", "false")
            .config("spark.sql.codegen.factoryMode", "NO_CODEGEN")
            .getOrCreate())
    
    def add_1minute_candle(self, candle: Dict):
        #adding 1m candles to the buffer, aggregating to 30m when there's 30 of them and adding to sliding window. unused in the final version
        self.minute_buffer.append(candle)
        
        if len(self.minute_buffer) >= 30:
            self._process_30minute_candle()
            self.minute_buffer.clear()
    
    def add_candle(self, candle: Dict):
        #adding candles to the system, 1m get buffered and aggregated, 30m get added directly
        interval = candle.get('interval', '1m')
        
        if interval == '30m':
            self.add_30minute_candle(candle)
        else:
            self.add_1minute_candle(candle)
    
    def add_30minute_candle(self, candle: Dict):
        #adding pre-aggregated 30m candles, used for bootstrapping
        #converts dicts to row-like objects for consistency
        from pyspark.sql import Row

        candle_row = Row(
            open_time=candle['open_time'],
            symbol=candle['symbol'],
            open=candle['open'],
            high=candle['high'],
            low=candle['low'],
            close=candle['close'],
            volume=candle['volume'],
            number_of_trades=candle['number_of_trades'],
            taker_buy_quote_asset_volume=candle['taker_buy_quote_asset_volume']
        )
        
        self.window_30m.append(candle_row)
        
        logger.info(
            f"Added 30m candle: {candle['symbol']} "
            f"@ {candle['open_time']} | Close: {candle['close']:.2f} "
            f"(Window: {len(self.window_30m)}/{WINDOW_SIZE})"
        )
        
        #generating preds when window is full
        if len(self.window_30m) == WINDOW_SIZE:
            self._predict()
    
    def _process_30minute_candle(self):
        #agg 1m candles
        df_1min = self.spark.createDataFrame(self.minute_buffer)
        df_30m = aggregate_candles(df_1min)
        #extract single row
        new_30m_row = df_30m.collect()[0]
        self.window_30m.append(new_30m_row)
        
        logger.info(
            f"Aggregated 30m candle: {new_30m_row['symbol']} "
            f"@ {new_30m_row['open_time']} | Close: {new_30m_row['close']:.2f}"
        )
        
        #generating preds
        if len(self.window_30m) == WINDOW_SIZE:
            self._predict()
    
    def _predict(self):
        #converts window to df, generates features without label, runs the models, extracts probabilities, runs a meta-learner, determines trading signal, backfills prediction label(not functional for now)
        df_window = self.spark.createDataFrame(list(self.window_30m))
        #getting the last candle in window data
        current_candle = self.window_30m[-1]
        
        # Backfill label for previous prediction if we have one
        # Now that a new candle has arrived, we know if the previous prediction was correct!
        if self.pending_prediction is not None:
            self._backfill_actual_label(
                self.pending_prediction['timestamp'],
                self.pending_prediction['close'],
                float(current_candle['close'])
            )
        
        df_features, _ = generate_features(
            df_window, 
            dataset_name="LIVE_STREAM",
            generate_label=False
        )
        
        if df_features.count() == 0:
            logger.warning("No features generated - skipping prediction")
            return
        
        rf_res = self.rf_model.transform(df_features).withColumn(
            "rf_prob", extract_prob_udf(col("probability"))
        )
        gbt_res = self.gbt_model.transform(df_features).withColumn(
            "gbt_prob", extract_prob_udf(col("probability"))
        )
        lr_res = self.lr_model.transform(df_features).withColumn(
            "lr_prob", extract_prob_udf(col("probability"))
        )
        
        meta_df = (rf_res.select("open_time", "symbol", "close", "rf_prob")
            .join(gbt_res.select("open_time", "gbt_prob"), "open_time")
            .join(lr_res.select("open_time", "lr_prob"), "open_time"))
        
        meta_assembler = VectorAssembler(
            inputCols=["rf_prob", "gbt_prob", "lr_prob"],
            outputCol="meta_features"
        )
        meta_df = meta_assembler.transform(meta_df)
        
        final_prediction = self.meta_model.transform(meta_df)
        final_prediction = final_prediction.withColumn(
            "final_prob", extract_prob_udf(col("probability"))
        )
        
        #extracting results
        result = final_prediction.collect()[0]
        prob_up = float(result['final_prob'])
        prediction = float(result['prediction'])
        
        #determining the signal based on confidence
        signal, confidence = self._determine_signal(prob_up, prediction)
        
        #displaying signal
        self._print_trade_signal(result['symbol'], signal, prob_up, confidence)
        
        #logging prediction
        self._log_prediction(result, prob_up, prediction, current_candle)
        
        logger.info(
            f"Prediction: {result['symbol']} | Prob UP: {prob_up:.4f} | "
            f"Signal: {signal} | Confidence: {confidence}"
        )
    
    def _backfill_actual_label(self, prev_timestamp: int, prev_close: float, current_close: float):
        #backfilling the label for previous prediction in CSV
        import csv
        
        try:
            #finding actual outcome
            actual_label = 1 if current_close > prev_close else 0
            
            #getting previous prediction
            predicted = self.pending_prediction.get('prediction', None)
            row_num = self.pending_prediction.get('row_number', None)
            
            if predicted is not None and row_num is not None:
                is_correct = 1 if predicted == actual_label else 0
                
                predicted_str = "UP" if predicted == 1 else "DOWN"
                actual_str = "UP" if actual_label == 1 else "DOWN"
                
                if is_correct == 1:
                    logger.info(f"Previous prediction correct: Predicted {predicted_str}, Actual {actual_str}")
                else:
                    logger.info(f"Previous prediction wrong: Predicted {predicted_str}, Actual {actual_str}")
                
                # Update the CSV file
                log_file = "data/predictions.csv"
                
                # Read all rows
                with open(log_file, 'r') as f:
                    rows = list(csv.reader(f))
                
                # Update the specific row (row_num is already accounting for header)
                if row_num < len(rows):
                    rows[row_num][6] = str(actual_label)  # actual_label column
                    rows[row_num][7] = str(is_correct)    # is_correct column
                
                # Write back
                with open(log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                
                logger.info(f"Backfilled row {row_num} in CSV")
                
        except Exception as e:
            logger.error(f"Error backfilling: {e}")
    
    def _determine_signal(self, prob_up: float, prediction: float) -> tuple:
        #determining the trading signal based on probability thresholds
        if prob_up >= 0.7 or prob_up <= 0.3:
            confidence = "VERY HIGH"
            signal = "BUY (Long)" if prediction == 1.0 else "SELL (Short)"
        elif prob_up >= 0.6 or prob_up <= 0.4:
            confidence = "HIGH"
            signal = "BUY (Long)" if prediction == 1.0 else "SELL (Short)"
        else:
            confidence = "Low"
            signal = "WAIT"
        
        return signal, confidence
    
    def _log_prediction(self, result, prob_up: float, prediction: float, current_candle):
        #logging preds to CSV file (append-only, no conflicts)
        import csv
        from datetime import datetime
        
        log_file = "data/predictions.csv"
        
        # Create file with header if it doesn't exist
        if not os.path.exists(log_file):
            os.makedirs("data", exist_ok=True)
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'datetime', 'symbol', 'close', 
                    'final_prob', 'prediction', 'actual_label', 'is_correct'
                ])
        
        # Append prediction (actual_label and is_correct will be updated later)
        timestamp = result['open_time']
        dt_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                dt_str,
                result['symbol'],
                float(result['close']),
                float(prob_up),
                int(prediction),
                '',  # actual_label - empty for now
                ''   # is_correct - empty for now
            ])
        
        logger.info(f"Logged prediction to {log_file}")
        
        # Store for backfilling in next cycle
        self.pending_prediction = {
            'timestamp': result['open_time'],
            'close': float(current_candle['close']),
            'prediction': int(prediction),
            'row_number': self._count_csv_rows(log_file)  # Track which row to update
        }
    
    def _count_csv_rows(self, filepath):
        """Count rows in CSV (excluding header)"""
        with open(filepath, 'r') as f:
            return sum(1 for line in f) - 1  # -1 for header
    
    def _print_trade_signal(self, symbol: str, signal: str, prob: float, confidence: str):
        #printing colored signals to console
        if "BUY" in signal:
            color = "\033[92m"  # Green
        elif "SELL" in signal:
            color = "\033[91m"  # Red
        else:
            color = "\033[93m"  # Yellow
        reset = "\033[0m"
        
        print(f"Signal: {symbol}")
        print(f"Action: {color}{signal}{reset}")
        print(f"Probability: {prob:.4f}")
        print(f"Confidence Level: {confidence}")


def run_stack_inference(parsed_streaming_df):
    #connects the Kafka stream to the CandlesProcessor and runs continuous inference
    processor = CandlesProcessor()
    
    def batch_function(batch_df, batch_id):
        #processing each micro-batch from stream
        if batch_df.count() == 0:
            return
        
        records = batch_df.collect()
        for row in records:
            processor.add_candle(row.asDict())  #auto detects 1m vs 30m
    
    #starting streaming query
    query = (parsed_streaming_df.writeStream
        .foreachBatch(batch_function)
        .option("checkpointLocation", "./binance_kline_chk")
        .trigger(processingTime="10 seconds")
        .start())
    
    logger.info("Stream started, waiting for candles")
    logger.info("Note: First prediction will have no backfilled prediction")
    logger.info("Subsequent predictions will show if previous prediction was correct")
    return query


if __name__ == "__main__":
    #standalone execution for testing
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, IntegerType
    
    spark = SparkSession.builder.appName("CryptoSignals").getOrCreate()
    
    schema = StructType([
        StructField("open_time", LongType(), True),
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", DoubleType(), True),
        StructField("quote_asset_volume", DoubleType(), True),
        StructField("number_of_trades", IntegerType(), True),
        StructField("taker_buy_base_asset_volume", DoubleType(), True),
        StructField("taker_buy_quote_asset_volume", DoubleType(), True),
        StructField("symbol", StringType(), True),
        StructField("interval", StringType(), True),
        StructField("ingested_at", StringType(), True)
    ])
    
    #reading from kafka
    df_stream = (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "binance_kline")
        .option("startingOffsets", "earliest")
        .load())
    
    #parsing json
    from pyspark.sql.functions import from_json
    
    df_parsed = df_stream.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")
    
    #running inference
    query = run_stack_inference(df_parsed)
    
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("Stopping the stream")
        query.stop()
