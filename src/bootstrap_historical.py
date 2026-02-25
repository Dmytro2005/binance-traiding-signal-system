#fetches the last 21 30-minutes candles from Binance, has to be ran once before starting the real-time inference
import requests
import time
from datetime import datetime, timezone
import json
import os
from kafka import KafkaProducer

#config
SYMBOLS = ["BTCUSDT"]  #fsymbol to fetch
INTERVAL = "30m"
LIMIT = 21  
KAFKA_TOPIC = "binance_kline"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

#API endpoint
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"


def fetch_historical_candles(symbol, interval="30m", limit=21):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    try:
        response = requests.get(BINANCE_API_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        candles = []
        
        for kline in data:
            candle = {
                "open_time": int(kline[0]), 
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "quote_asset_volume": float(kline[7]),
                "number_of_trades": int(kline[8]),
                "taker_buy_base_asset_volume": float(kline[9]),
                "taker_buy_quote_asset_volume": float(kline[10]),
                "symbol": symbol,
                "interval": interval,
                "ingested_at": datetime.now(timezone.utc).isoformat()
            }
            candles.append(candle)
        
        return candles
        
    except Exception as e:
        print(f"Error fetching candles for {symbol}: {e}")
        return []


def send_to_kafka(candles, producer):
    #sending candles to Kafka topic
    for candle in candles:
        producer.send(KAFKA_TOPIC, value=candle)
        print(f"Sent: {candle['symbol']} @ {candle['open_time']} | Close: {candle['close']:.2f}")


def save_to_json(candles, filename):
    #saving to a JSON file as a backup
    with open(filename, 'w') as f:
        json.dump(candles, f, indent=2)
    print(f"Saved to {filename}")


def bootstrap_system(use_kafka=True, save_json=True):
    #initializing producer if needed
    producer = None
    if use_kafka:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all'
            )
            print("✓ Connected to Kafka")
        except Exception as e:
            print(f"✗ Kafka connection failed: {e}")
            print("  Falling back to JSON-only mode")
            use_kafka = False
    
    print()
    
    #fetching and processing candles for each symbol
    all_candles = {}
    
    for symbol in SYMBOLS:
        print(f"Fetching {symbol}")
        candles = fetch_historical_candles(symbol, INTERVAL, LIMIT)
        
        if candles:
            all_candles[symbol] = candles
            
            #sending to kafka
            if use_kafka and producer:
                send_to_kafka(candles, producer)
            
            #saving to JSON
            if save_json:
                os.makedirs("data/bootstrap", exist_ok=True)
                filename = f"data/bootstrap/{symbol}_{INTERVAL}_bootstrap.json"
                save_to_json(candles, filename)
            
            print()
            time.sleep(0.5)  # Rate limiting
        else:
            print(f"Failed to fetch candles")
            print()
    
    #cleanup
    if producer:
        producer.flush()
        producer.close()
        print("Producer closed")
    
    print("Bootstrap complete")
    print("Summary:")
    print(f"Symbols bootstrapped: {len(all_candles)}/{len(SYMBOLS)}")
    print(f"Total candles: {sum(len(c) for c in all_candles.values())}")
    
    
    return all_candles


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bootstrap with historical candles")
    parser.add_argument("--no-kafka", action="store_true", help="Don't send to Kafka, only save JSON")
    parser.add_argument("--no-json", action="store_true", help="Don't save JSON, only send to Kafka")
    
    args = parser.parse_args()
    
    use_kafka = not args.no_kafka
    save_json = not args.no_json
    
    bootstrap_system(use_kafka=use_kafka, save_json=save_json)
