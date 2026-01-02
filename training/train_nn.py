import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
import mlflow
import mlflow.pytorch
import numpy as np

from utils.eval import log_metrics_to_mlflow


# load data
df = pd.read_parquet("data/processed/processed_df.parquet")
X = df.drop(columns=["Class"]).values
y = df["Class"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y_val if False else y  # stratify=y
)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.float32)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)

# Define model
class FraudNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FraudNN(X_train_t.shape[1]).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Fraud Detection Experiment")
with mlflow.start_run(run_name="Torch_Classifier_FraudNN"):
    mlflow.log_param("model", "FraudNN_Torch")
    mlflow.log_param("learning_rate", 1e-3)
    mlflow.log_param("batch_size", 256)
    mlflow.log_param("epochs", 10)

    # Training loop
    for epoch in range(10):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).view(-1, 1)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.5f}")

    # Validation metrics
    model.eval()
    val_probs = []
    val_labels = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            probs = model(xb).cpu().numpy()
            val_probs.extend(probs)
            val_labels.extend(yb.numpy())

    val_probs = np.array(val_probs).flatten()
    ap = average_precision_score(y_val, val_probs)
    auc = roc_auc_score(y_val, val_probs)
    val_pred_labels = (val_probs >= 0.5).astype(int)
    tn, fp, fn, tp = pd.crosstab(y_val, val_pred_labels).values.ravel()
    recall = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    fbeta = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall + 1e-9)

    # Log metrics to mlflow
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

    # Log model
    mlflow.pytorch.log_model(model, "fraudnn_model")

# Save model locally too
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/fraudnn_torch.pt")
print("Torch classifier model saved to models/fraudnn_torch.pt")