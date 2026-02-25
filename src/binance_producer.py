#Websocket producer,connects to Binance's WebSocket API and streams closed candles to a Kafka topic

import json
import asyncio
import websockets
from kafka import KafkaProducer
from datetime import datetime, timezone
from typing import Dict, List, Optional
import logging

#logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceKlineProducer:
    def __init__(
        self,
        symbols: List[str],
        kafka_topic: str = "binance_kline",
        kafka_bootstrap_servers: str = "localhost:9092",
        interval: str = "30m"
    ):
        #initializing 
        
        self.symbols = [s.lower() for s in symbols]
        self.kafka_topic = kafka_topic
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.interval = interval
        
        #validating the interval, 1m candles are proving to be an issue, since to build a proper 30m candle out of them the streaming has to start at exactly X:00 or X:30, so sticking with 30m
        if interval not in ["1m", "30m"]:
            raise ValueError("Interval must be '1m' or '30m'")
        
        # Build WebSocket URL for multiple symbols
        streams = "/".join([f"{symbol}@kline_{interval}" for symbol in self.symbols])
        self.ws_url = f"wss://stream.binance.com/stream?streams={streams}"
        
        #init kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',  # Wait for all replicas
            retries=3
        )
        
        logger.info(f"Initialized producer for {len(symbols)} symbols: {', '.join(self.symbols)}")
        logger.info(f"WebSocket URL: {self.ws_url}")
        logger.info(f"Kafka topic: {kafka_topic}")
    
    def parse_kline(self, message: Dict) -> Optional[Dict]:
        #parsing a websocket message and extracting closed candle data, matching the training schema 
        try:
            #handling combined stream format
            data = message.get("data", message)
            k = data.get("k")
            
            if not k:
                return None
            
            #only processing closed candles (x=True)
            if not k.get("x"):
                return None
            
            #extracting the data
            record = {
                "open_time": k["t"],  #unix timestamp but shouldn't be an issue
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
                "quote_asset_volume": float(k["q"]),
                "number_of_trades": int(k["n"]),
                "taker_buy_base_asset_volume": float(k["V"]),
                "taker_buy_quote_asset_volume": float(k["Q"]),
                "symbol": data.get("s", "").upper(),  #uppercase for consistency
                "interval": k["i"],
                "ingested_at": datetime.now(timezone.utc).isoformat()
            }
            
            return record
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing {e}")
            return None
    
    async def run(self):
        #main producer loop, connects to websocket and forwards candles to kafka
        logger.info("Starting the producer")
        
        while True:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10
                ) as ws:
                    logger.info(f"Connected to websocket for {len(self.symbols)} symbols")
                    
                    candle_count = 0
                    async for raw_message in ws:
                        message = json.loads(raw_message)
                        record = self.parse_kline(message)
                        
                        if record:
                            #sending to kafka
                            self.producer.send(self.kafka_topic, value=record)
                            candle_count += 1
                            
                            logger.info(
                                f"Sent candle #{candle_count}: {record['symbol']} "
                                f"@ {record['open_time']} | Close: {record['close']:.2f}"
                            )
                            
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}, reconnecting")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Unexpected error: {e}, reconnecting")
                await asyncio.sleep(5)
    
    def close(self):
        #cleanup
        self.producer.flush()
        self.producer.close()
        logger.info("Producer closed")


async def main():
    #entry point
    #config
    SYMBOLS = ["BTCUSDT"]
    KAFKA_TOPIC = "binance_kline"
    KAFKA_SERVERS = "localhost:9092"
    
    producer = BinanceKlineProducer(
        symbols=SYMBOLS,
        kafka_topic=KAFKA_TOPIC,
        kafka_bootstrap_servers=KAFKA_SERVERS,
        interval="30m"
    )
    
    try:
        await producer.run()
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        producer.close()


if __name__ == "__main__":
    asyncio.run(main())
