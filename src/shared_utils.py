#contains functions used by training and inference, only imported to inference, included in case we have enough time to slighly rewrite the training markdown file for better structure
from pyspark.sql import Window
from pyspark.sql.functions import (
    col, lag, lead, when, unix_timestamp, floor, lit,
    first, last, max as _max, min as _min, sum as _sum,
    pow, sqrt, avg, abs as sabs, greatest, udf
)
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from functools import reduce


#UDF to extract probability of UP class from the probability vectore
extract_prob_udf = udf(lambda v: float(v[1]) if v is not None else 0.5, DoubleType())


def aggregate_candles(df, candle_minutes=30):
    #aggregates the cm candles into larger timeframes
    from pyspark.sql.types import TimestampType

    open_time_type = df.schema["open_time"].dataType
    
    #converts to unix ms  if it's a timestamp
    if isinstance(open_time_type, TimestampType):
        df = df.withColumn("open_time_ms", unix_timestamp(col("open_time")) * 1000)
    else:
        #streaming - already unix
        df = df.withColumn("open_time_ms", col("open_time"))
    
    df = df.withColumn(
        "time_bucket",
        floor(col("open_time_ms") / lit(candle_minutes * 60 * 1000)) 
        * lit(candle_minutes * 60 * 1000)
    )
    
    df = df.groupBy("symbol", "time_bucket").agg(
        first("open").alias("open"),
        _max("high").alias("high"),
        _min("low").alias("low"),
        last("close").alias("close"),
        _sum("volume").alias("volume"),
        _sum("number_of_trades").alias("number_of_trades"),
        _sum("taker_buy_quote_asset_volume").alias("taker_buy_quote_asset_volume")
    ).withColumnRenamed("time_bucket", "open_time").orderBy("symbol", "open_time")
    
    return df


def generate_features(df, dataset_name="", lookback=20, generate_label=True):
    print(f"Start feature generation for {dataset_name}")
    
    window_spec = Window.partitionBy("symbol").orderBy("open_time")
    window_symbol = Window.partitionBy("symbol").orderBy("open_time")
    
    #used lag features
    lag_config = {
        "high": [1, 2, 3, 4, 5, 6],
        "close": list(range(1, lookback + 1)),
        "open": [3, 5, 6],
        "number_of_trades": [1, 2, 3],
        "taker_buy_quote_asset_volume": [1, 2, 3]
    }
    
    #creating lag
    for column, lags in lag_config.items():
        for lag_period in lags:
            df = df.withColumn(
                f"{column}_lag{lag_period}",
                lag(col(column), lag_period).over(window_spec)
            )
    
    #generating label, only used for training
    if generate_label:
        df = df.withColumn("close_next", lead(col("close"), 1).over(window_spec))
        df = df.withColumn("label",
            when(col("close_next") > col("close"), 1)
            .when(col("close_next") < col("close"), 0)
            .otherwise(0)
        )
        df = df.drop("close_next")
    else:
        #inference - creates a dummy label that's not used for predictions since the model expects it to exist
        df = df.withColumn("label", lit(0))  # Dummy value, ignored during inference
    
    #derived features
    df = df.withColumn("body", col("close") - col("open"))
    df = df.withColumn("range", col("high") - col("low"))
    df = df.withColumn(
        "upper_wick",
        col("high") - when(col("close") > col("open"), col("close")).otherwise(col("open"))
    )
    df = df.withColumn(
        "lower_wick",
        when(col("close") < col("open"), col("close")).otherwise(col("open")) - col("low")
    )
    
    #smas
    df = df.withColumn("sma_5",
        (col("close_lag1") + col("close_lag2") + col("close_lag3") + col("close_lag4") + col("close_lag5")) / 5)
    df = df.withColumn("price_to_sma5",
        when(col("sma_5") != 0, (col("close") - col("sma_5")) / col("sma_5")).otherwise(0))
    df = df.drop("sma_5")

    close_lags_10 = [col(f"close_lag{i}") for i in range(1, 11)]
    df = df.withColumn("sma_10", reduce(lambda a, b: a + b, close_lags_10) / lit(10))
    df = df.withColumn("price_to_sma10",
        when(col("sma_10") != 0, (col("close") - col("sma_10")) / col("sma_10")).otherwise(0)
    )
    df = df.drop("sma_10")

    close_lags_20 = [col(f"close_lag{i}") for i in range(1, 21)]
    df = df.withColumn("sma_20", reduce(lambda a, b: a + b, close_lags_20) / lit(20))
    df = df.withColumn("price_to_sma20",
        when(col("sma_20") != 0, (col("close") - col("sma_20")) / col("sma_20")).otherwise(0)
    )

    df = df.withColumn("price_momentum",
        when(col("close_lag5") != 0, (col("close") - col("close_lag5")) / col("close_lag5"))
        .otherwise(0)
    )
    
    df = df.withColumn("volatility",
        sqrt((
            pow((col("close_lag1") - col("close_lag2")) / col("close_lag2"), 2) +
            pow((col("close_lag2") - col("close_lag3")) / col("close_lag3"), 2) +
            pow((col("close_lag3") - col("close_lag4")) / col("close_lag4"), 2)
        ) / 3)
    )
    
    df = df.withColumn("bb_position",
        when(col("volatility") != 0,
            (col("close") - col("sma_20")) / (2 * col("volatility"))
        ).otherwise(0)
    )
    df = df.drop("volatility", "sma_20")
    
    df = df.withColumn("true_range",
        greatest(
            col("high") - col("low"),
            sabs(col("high") - lag(col("close"), 1).over(window_symbol)),
            sabs(col("low") - lag(col("close"), 1).over(window_symbol))
        )
    )
    window_tr = Window.partitionBy("symbol").orderBy("open_time").rowsBetween(-10, -1)
    df = df.withColumn("atr_10", avg(col("true_range")).over(window_tr))
    df = df.drop("true_range")
    
    df = df.dropna()
    
    #selecting best performing features
    SELECTED_FEATURES = [
        #sma features
        "price_to_sma5", "price_to_sma10", "price_to_sma20",
        "price_momentum",
        
        #candle patterns
        "body", "range", "upper_wick", "lower_wick",
        
        #lags
        "high_lag1", "high_lag2", "high_lag3", "high_lag4", "high_lag5", "high_lag6",
        "close_lag3", "close_lag4",
        "open_lag3", "open_lag5", "open_lag6",
        "number_of_trades_lag1", "number_of_trades_lag2", "number_of_trades_lag3",
        "taker_buy_quote_asset_volume_lag1",
        "taker_buy_quote_asset_volume_lag2",
        "taker_buy_quote_asset_volume_lag3",
        
        #advanced
        "bb_position", "atr_10"
    ]
    
    #dropping unused columns to reduce memory
    system_cols = ["symbol", "open_time", "label", "close"]
    keep_cols = set(SELECTED_FEATURES + system_cols)
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        df = df.drop(*drop_cols)
    
    #feature vector
    assembler = VectorAssembler(
        inputCols=SELECTED_FEATURES,
        outputCol="features",
        handleInvalid="skip"
    )
    df = assembler.transform(df)
    
    #final columns, keeping close for display in predictions
    result_df = df.select("symbol", "features", "label", "open_time", "close")
    
    return result_df, SELECTED_FEATURES
