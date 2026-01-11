import joblib
import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import mlflow
import mlflow.sklearn

from utils.eval import log_metrics_to_mlflow

df = pd.read_parquet("data/processed/processed_df.parquet")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
exp = mlflow.set_experiment("Fraud Detection Experiment")
with mlflow.start_run(run_name="XGBoost_Fraud_Model", experiment_id=exp.experiment_id):
    model = XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric="auc"
    )
    model.fit(X_train, y_train)

    y_pred_label = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]

    ap_score = average_precision_score(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)
    tn, fp, fn, tp = pd.crosstab(y_test, y_pred_label).values.ravel()
    recall = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    fbeta = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall + 1e-9)

    # Log metrics to mlflow
    log_metrics_to_mlflow(
        roc_auc=auc_score,
        pr_auc=ap_score,
        recall=recall,
        precision=precision,
        fbeta=fbeta,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp
    )

    # Log model to mlflow
    mlflow.sklearn.log_model(model, name="xgboost_model")


# Save the trained model locally
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model_xgb.pkl")
print("XGBoost model training complete. Model saved to models/model_xgb.pkl")