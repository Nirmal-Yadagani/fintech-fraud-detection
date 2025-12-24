from typing import Dict

from fastapi import FastAPI
import pandas as pd
import lightgbm as lgb
from pydantic import BaseModel
import mlflow
import psycopg2

from features.feature_defs import build_features

app = FastAPI(title='FinTech Fraud Scoring API')

# load model from mlflow export
model = mlflow.lightgbm.load_model(model_uri='mlruns/1/models/m-02dc49dabf354f86833888f88dc95e74/artifacts/')


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


def fetch_transaction() -> Dict:
    conn = psycopg2.connect(dbname='postgres', user='postgres')
    cursor = conn.cursor()
    cursor.execute("""FROM online_tx ORDER BY Time DESC LIMIT 1;""")
    record = cursor.fetchone()
    cursor.close()
    conn.close()

    if not record:
        raise ValueError("Transaction not found in the database.")

    cols = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
    return dict(zip(cols, record))

@app.post('/score')
def score_tx(tx: Transaction):
    df = pd.DataFrame([dict(tx)])
    df.drop(columns=['tx_uid'], inplace=True)
    df = build_features(df)
    score = model.predict(df)[0]

    return {'fraud_risk_score': float(score),
            'fraud_label': int(score > 0.7),
            }

@app.post('/score_db')
def score_tx_db():
    tx = fetch_transaction()
    if not tx:
        return {'error': 'No transaction found in the database.'}
    df = pd.DataFrame([tx])
    df = build_features(df)
    score = model.predict(df)[0]
    return {'fraud_risk_score': float(score),
            'fraud_label': int(score > 0.7),
            }


@app.get('/')
def root():
    return {'message': 'Welcome to the FinTech Fraud Scoring API!'}
