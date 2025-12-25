#!/bin/bash
set -e

echo "Starting Zookeeper..."
./kafka_2.13-3.8.0/bin/zookeeper-server-start.sh ./kafka_2.13-3.8.0/config/zookeeper.properties > zk.log 2>&1 &
sleep 5

echo "Starting Kafka broker..."
./kafka_2.13-3.8.0/bin/kafka-server-start.sh ./kafka_2.13-3.8.0/config/server.properties > kafka.log 2>&1 &
sleep 5

echo "Creating topic (safe path for prod clusters)..."
./kafka_2.13-3.8.0/bin/kafka-topics.sh --create \
  --topic tx_stream \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists
sleep 2

echo "Setting up Postgres table..."
sudo -u postgres psql -c "
CREATE TABLE IF NOT EXISTS online_tx (
    Time FLOAT,
    $(for i in {1..28}; do echo -n "V$i FLOAT, "; done)
    Amount FLOAT
);
"

echo "Installing Python dependencies..."
pip install -q kafka-python fastapi uvicorn lightgbm psycopg2-binary numpy pandas

echo "Exporting model if not already exported..."
python3 - <<EOF
import lightgbm as lgb
m = lgb.Booster(model_file="training/model.txt")
print("Model loaded for inference:", m)
EOF

echo "Starting Kafka consumer service (writes stream to Postgres)..."
python3 streaming/consumer.py > consumer.log 2>&1 &
sleep 3

echo "Starting Kafka producer service (generates transactions)..."
python3 streaming/producer.py > producer.log 2>&1 &
sleep 3

echo "Starting FastAPI fraud scoring API..."
uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload > api.log 2>&1 &
sleep 5

echo "Verifying services..."
jps  # shows Java processes like Kafka, Zookeeper
echo "Kafka topics on broker:"
./kafka_2.13-3.8.0/bin/kafka-topics.sh --list --bootstrap-server localhost:9092

echo "API status:"
curl -s http://localhost:8000/ | jq || echo "API not responding!!!"

echo "Deployment completed!"