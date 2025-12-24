import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import mlflow

from features.feature_defs import build_features

mlflow.set_experiment("Fraud Detection Experiment")

df = pd.read_csv('data/raw/creditcard.csv')
df = build_features(df)

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

with mlflow.start_run():
    mlflow.log_params(params)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500
    )

    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    ap_score = average_precision_score(y_val, y_val_pred)

    mlflow.log_metric("val_ap", ap_score)
    mlflow.lightgbm.log_model(model, name="lightgbm_model")

    print(f"Validation Average Precision: {ap_score:.4f}")