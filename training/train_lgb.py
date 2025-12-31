import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
import mlflow


mlflow.set_experiment("Fraud Detection Experiment")

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

mlflow.set_tracking_uri("mlruns")
exp = mlflow.set_experiment("Fraud Detection Experiment")
with mlflow.start_run("LightGBM_Fraud_Model", experiment_id=exp.experiment_id):
    mlflow.log_params(params)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500
    )

    y_pred_label = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_prob = model.predict_proba(X_val)[:,1]

    ap_score = average_precision_score(y_val, y_pred_prob)
    auc_score = roc_auc_score(y_val, y_pred_prob)
    clf_report = classification_report(y_val, y_pred_label)


    mlflow.log_metric("val_ap", ap_score)
    mlflow.log_metric("val_auc", auc_score)
    mlflow.log_text("clf_report", clf_report)
    mlflow.lightgbm.log_model(model, name="lightgbm_model")

    print(f"Validation Average Precision: {ap_score:.4f}")
    print(f"Validation Roc Auc Score: {auc_score:.4f}")
    print(f"Validation Classification Report: {clf_report}")

    model.save_model('training/model_lgb.txt')