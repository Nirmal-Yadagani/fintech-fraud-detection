import json
import time
import random
import numpy as np
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable


BOOTSTRAP = "kafka:9092" if "docker" in __file__.lower() else "localhost:9092"

# Retry until Kafka is ready
while True:
    try:
        producer = KafkaProducer(bootstrap_servers=BOOTSTRAP, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        print("Kafka is ready! Producer connected.")
        break
    except NoBrokersAvailable:
        print("Waiting for Kafka broker...")
        time.sleep(3)

def generate_transaction():
    transaction = {
        "Time": float(time.time()),  # unix timestamp
        **{f"V{i}": random.uniform(-5, 5) for i in range(1, 29)},
        "Amount": float(np.random.lognormal(mean=8, sigma=1.2))
    }
    return transaction


print("Starting transaction producer...")
while True:
    tx = generate_transaction()
    producer.send('transactions', value=tx)
    # print(f"Produced transaction: {tx}")
    time.sleep(random.uniform(0.2, 1.5))  # Simulate a delay between transactions