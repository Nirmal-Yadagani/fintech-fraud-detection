#!/bin/bash

KAFKA_HOME="kafka"
TOPIC="transactions"
PARTITIONS=3
REPLICATION=1

echo "🧠 Creating Kafka topic: $TOPIC"

$KAFKA_HOME/bin/kafka-topics.sh \
  --create \
  --topic $TOPIC \
  --bootstrap-server localhost:9092 \
  --partitions $PARTITIONS \
  --replication-factor $REPLICATION \
  --if-not-exists

echo "Topic created and ready for use!"
