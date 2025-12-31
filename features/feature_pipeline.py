import pandas as pd
from .feature_defs import build_features

if __name__ == "__main__":
    df = pd.read_csv("data/raw/creditcard.csv")
    df["Time"] = df["Time"].astype(float)

    df_feat = build_features(df)
    df_feat.to_parquet("data/processed/processed_df.parquet", index=False)
    print("Processed features and saved to data/processed/processed_df.parquet")
