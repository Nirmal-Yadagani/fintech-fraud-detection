import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['hour'] = (df['Time']//3600)%24
    df['is_night'] = df['hour'].between(0,5)

    df['amount_log'] = np.log1p(df['Amount'].clip(0)+1)

    return df.drop(columns=['Amount','Time'])