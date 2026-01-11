import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
import mlflow

from utils.eval import log_metrics_to_mlflow

df = pd.read_parquet('data/processed/processed_df.parquet')

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "binary",
    "metric": "aucpr",
    "boosting_type": "gbdt",
    "device": "gpu",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "verbose": 100
}

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Fraud Detection Experiment")
with mlflow.start_run(run_name="LightGBM_Fraud_Model"):
    mlflow.log_params(params)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500
    )

    y_pred_label = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_prob = model.predict(X_val, raw_score=True)
    tn, fp, fn, tp = pd.crosstab(y_val, (y_pred_prob > 0.5).astype(int)).values.ravel()
    ap_score = average_precision_score(y_val, y_pred_prob)
    auc_score = roc_auc_score(y_val, y_pred_prob)
    ap = ap_score
    auc = auc_score
    recall = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    fbeta = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall + 1e-9)

    # log metrics to MLflow
    log_metrics_to_mlflow(
        roc_auc=auc,
        pr_auc=ap,
        recall=recall,
        precision=precision,
        fbeta=fbeta,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp
    )

    # Log model to MLflow
    mlflow.lightgbm.log_model(model, name="lightgbm_model")


# Save the trained model locally
os.makedirs("models", exist_ok=True)
model.save_model('models/model_lgb.txt')
print("LightGBM model training complete. Model saved to models/model_lgb.txt")