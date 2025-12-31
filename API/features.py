import pandas as pd
import numpy as np
import joblib

# 1. Load your new comprehensive metadata
METADATA = joblib.load(r"C:\Users\user\ml projects\fraud detection project\notebooks\lgb_feature_metadata.joblib")
FEATURE_COLUMNS = METADATA["feature_names"]
CATEGORICAL_FEATURES = METADATA["categorical_features"]
CATEGORICAL_MAPPINGS = METADATA["categorical_mappings"]

# Load other artifacts
MEDIANS = joblib.load("training_medians.joblib")
CARD1_AMT_MEANS = joblib.load("card1_amt_means.joblib")
UID_AMT_MEANS = joblib.load("uid_amt_means.joblib")
UID_D1_MEANS = joblib.load("uid_d1_means.joblib")
UID_D15_MEANS = joblib.load("uid_d15_means.joblib")
C_FEATS_MEANS = joblib.load("c_feats_card1_means.joblib")
CARD1_ADDR_NUNIQUE = joblib.load("card1_addr_nunique.joblib")

def preprocess_transaction(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # --- Behavioral Feature Engineering (same as yours) ---
    card1 = df["card1"].iloc[0]
    addr1 = df["addr1"].iloc[0] if "addr1" in df.columns else np.nan
    uid = (card1, addr1)

    df["card1_amt_mean"] = CARD1_AMT_MEANS.get(card1, np.nan)
    df["card1_amt_ratio"] = df["TransactionAmt"] / (df["card1_amt_mean"] + 0.01)
    df["uid_amt_mean"] = UID_AMT_MEANS.get(uid, np.nan)
    df["uid_amt_ratio"] = df["TransactionAmt"] / (df["uid_amt_mean"] + 0.01)
    df["card1_addr1_count"] = CARD1_ADDR_NUNIQUE.get(card1, 1)
    df["uid_D1_mean"] = UID_D1_MEANS.get(uid, np.nan)
    df["uid_D15_mean"] = UID_D15_MEANS.get(uid, np.nan)

    # --- Ensure all categorical columns exist ---
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = 'missing'  # create missing column if absent
        allowed = CATEGORICAL_MAPPINGS[col]
        if 'missing' not in allowed:
            allowed = allowed + ['missing']
        df[col] = df[col].astype(str).replace('nan', 'missing')
        df.loc[~df[col].isin(allowed), col] = 'missing'
        df[col] = pd.Categorical(df[col], categories=allowed)

    # --- Final schema alignment ---
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0  # numeric fill for missing engineered columns

    df = df[FEATURE_COLUMNS]

    # Fill remaining NaNs with training medians
    for col, median in MEDIANS.items():
        if col in df.columns:
            df[col] = df[col].fillna(median)

    # Ensure numeric types
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].astype('float32')

    return df
