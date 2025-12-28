import json, time
import pandas as pd
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import psycopg2
import uuid
from features.feature_defs import build_features
import lightgbm as lgb
import os

model = lgb.Booster(model_file='training/model.txt')

EXPECTED_SCHEMA = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

# Retry until Kafka is ready
while True:
    try:
        consumer = KafkaConsumer("transactions", bootstrap_servers=BOOTSTRAP,value_deserializer=lambda v: json.loads(v.decode('utf-8')))
        print("Kafka is ready! Consumer connected.")
        break
    except NoBrokersAvailable:
        print("Waiting for Kafka broker...")
        time.sleep(3)

conn = psycopg2.connect(dbname='frauddb',user='postgres',password='postgres',host='postgres',port=5432)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS online_tx;")
cursor.execute("""CREATE TABLE IF NOT EXISTS online_tx (
    tx_uid UUID PRIMARY KEY,
    Time FLOAT,
    V1 FLOAT,
    V2 FLOAT,
    V3 FLOAT,
    V4 FLOAT,
    V5 FLOAT,
    V6 FLOAT,
    V7 FLOAT,
    V8 FLOAT,
    V9 FLOAT,
    V10 FLOAT,
    V11 FLOAT,
    V12 FLOAT,
    V13 FLOAT,
    V14 FLOAT,
    V15 FLOAT,
    V16 FLOAT,
    V17 FLOAT,
    V18 FLOAT,
    V19 FLOAT,
    V20 FLOAT,
    V21 FLOAT,
    V22 FLOAT,
    V23 FLOAT,
    V24 FLOAT,
    V25 FLOAT,
    V26 FLOAT,
    V27 FLOAT,
    V28 FLOAT,
    Amount FLOAT
);
""")

conn.commit()

print("Started consuming & validating transactions...")
for tx in consumer:
    transaction = tx.value

    if sorted(transaction.keys()) != sorted(EXPECTED_SCHEMA):
        print(f"Invalid transaction schema: {transaction}")
        continue

    df = pd.DataFrame([transaction])[EXPECTED_SCHEMA]
    features = build_features(df)

    score = float(model.predict(features)[0])
    fraud = int(score > 0.7)

    transaction['tx_uid'] = str(uuid.uuid4())
    cols = ["tx_uid","Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
        "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22",
        "V23","V24","V25","V26","V27","V28","Amount"]
    values = tuple(transaction[c] for c in cols)
    cur = conn.cursor()

    cursor.execute("""
        INSERT INTO online_tx (tx_uid, Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14,
                            V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount)
        VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (tx_uid) DO NOTHING;
    """, values)
    conn.commit()
    
    print("Stored | Score:", round(score,4), "| Fraud:", fraud)