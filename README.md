# binance-traiding-signal-system
Real-time crypto trading signal pipeline: ingest 30-minute Binance candlesticks via WebSocket and Kafka, maintain a sliding window of 21 candles, run a stacked ML ensemble (Random Forest, GBT, Logistic Regression + meta-learner), and output **BUY** / **SELL** / **WAIT** signals with confidence.
