import pandas as pd
import numpy as np

def build_features(df):
    df = df.copy()
    df = df.sort_values("Time")

    # existing
    df['hour'] = (df['Time'] // 3600) % 24
    df['is_night'] = df['hour'].between(0, 5).astype(int)
    df['amount_log'] = np.log1p(df['Amount'].clip(0))

    # new derived
    df["delta_t"] = df["Time"].diff().fillna(0)

    df["tx_count_30s"] = df["Amount"].rolling(30).count().fillna(0)
    df["tx_count_1min"] = df["Amount"].rolling(60).count().fillna(0)
    df["tx_count_5min"] = df["Amount"].rolling(300).count().fillna(0)

    df["amt_sum_1min"] = df["Amount"].rolling(60).sum().fillna(0)
    df["amt_sum_5min"] = df["Amount"].rolling(300).sum().fillna(0)

    df["amt_mean_5"] = df["Amount"].rolling(5).mean().fillna(df["Amount"])
    df["amt_std_5"] = df["Amount"].rolling(5).std().fillna(0)
    df["amt_dev"] = df["Amount"] - df["amt_mean_5"]
    df["amt_zscore"] = df["amt_dev"] / (df["amt_std_5"] + 1e-6)

    v_cols = [f"V{i}" for i in range(1, 29)]
    df["v_std_5"] = df[v_cols].rolling(5).std().mean(axis=1).fillna(0)

    # Keep Amount for ops, drop only Time and raw Amount at end
    return df.drop(columns=["Amount", "Time"])
