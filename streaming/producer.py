import json
import time
import random
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


def generate_transaction():
    transaction = {
        "Time": time.time(),  # unix timestamp
        **{f"V{i}": random.uniform(-5, 5) for i in range(1, 29)},
        "Amount": round(random.expovariate(1/5000), 2)
    }
    return transaction


print("Starting transaction producer...")
while True:
    tx = generate_transaction()
    producer.send('transactions', value=tx)
    print(f"Produced transaction: {tx}")
    time.sleep(random.uniform(0.2, 1.5))  # Simulate a delay between transactions