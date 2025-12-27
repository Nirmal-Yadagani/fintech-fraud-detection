import json, time, random, os, socket
import numpy as np
from kafka import KafkaProducer, KafkaAdminClient
from kafka.errors import KafkaError

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC = "transactions"

# --- Now start producer ---
producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# ensure_topic()

def generate_tx():
    tx = {f"V{i}": float(random.gauss(0,1)) for i in range(1,29)}
    tx["Time"] = float(time.time())
    tx["Amount"] = float(np.random.lognormal(8,1.2))
    return tx

print("📤 Starting transaction producer stream...")

while True:
    tx = generate_tx()
    try:
        producer.send(TOPIC, tx)
        print("Sent:", tx)
    except KafkaError as e:
        print("❌ Kafka send error:", e)
    time.sleep(1)
