import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch
import os

# Load data
df = pd.read_parquet("data/processed/processed_df.parquet")

scale_cols = ["amount_log", "delta_t", "tx_count_30s", "tx_count_1min", "tx_count_5min",
              "amt_sum_1min", "amt_sum_5min", "amt_mean_5", "amt_std_5", "amt_dev",
              "amt_zscore", "v_std_5"]

# Scale features
scaler = StandardScaler()
non_fraud = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]

# Define train and test sets
X_train = non_fraud.drop(columns=['Class']).copy()
X_test = fraud.drop(columns=['Class']).copy()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

total_df = df.copy()
total_df[scale_cols] = scaler.transform(total_df[scale_cols])

# Convert to tensors
train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
val_tensor = torch.tensor(X_test.values, dtype=torch.float32)

train_ds = TensorDataset(train_tensor, train_tensor)
val_ds = TensorDataset(val_tensor, val_tensor)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

# Autoencoder Model
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoEncoder(train_tensor.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Fraud Detection Experiment")
with mlflow.start_run(run_name="Torch_AutoEncoder_Anomaly"):
    mlflow.log_param("model_type", "autoencoder_torch")
    mlflow.log_param("lr", 1e-3)
    mlflow.log_param("batch_size", 256)
    mlflow.log_param("epochs", 10)

    # Training Loop
    for epoch in range(1,50):
        model.train()
        losses = []
        for batch, _ in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            optimizer.zero_grad()
            loss = criterion(batch, recon)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        mlflow.log_metric("train_mse", avg_loss, step=epoch)
        print(f"Epoch {epoch} | Train MSE: {avg_loss:.5f}")

    # Log model
    mlflow.pytorch.log_model(model, name="autoencoder_model")

# ---- Generate anomaly scores on full dataset ----
model.eval()
full_tensor = torch.tensor(total_df.drop(columns=['Class']).values, dtype=torch.float32).to(device)
with torch.no_grad():
    recon_full = model(full_tensor)

mse = torch.mean((full_tensor - recon_full) ** 2, dim=1).cpu().numpy()
df["anomaly_score"] = mse

# Save processed data
os.makedirs("data/processed", exist_ok=True)
df.to_parquet("data/processed/processed_with_autoencoder.parquet", index=False)
print("Torch Autoencoder training complete. Anomaly scores saved!")
