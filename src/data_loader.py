import pandas as pd
from config import ROLLING_WINDOW


def load_and_preprocess_data(filepath="../data/Final_Fused_IoT_Dataset_FIXED.csv"):

    print("Loading from:", filepath)
    df = pd.read_csv(filepath)

    # Drop metadata/label columns not used as features
    df = df.drop(columns=["type", "final_label", "device"], errors="ignore")

    # 1️⃣ Sort by timestamp (CRITICAL)

    # Combine date and time into a real timestamp
    df["timestamp"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), format="mixed")

    # Sort chronologically
    df = df.sort_values(by="timestamp")

    # Drop original date/time columns
    df = df.drop(columns=["date", "time"])

    # 2️⃣ Create rolling temporal features
    window = ROLLING_WINDOW

    for col in ["temperature", "humidity", "pressure"]:
        df[f"{col}_roll_mean"] = df[col].rolling(window).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window).std()

    for col in ["latitude", "longitude"]:
        df[f"{col}_roll_mean"] = df[col].rolling(window).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window).std()

    # 3️⃣ Drop timestamp (after ordering)
    df = df.drop(columns=["timestamp"])

    # 4️⃣ Fill NaN from rolling
    df = df.fillna(0)

    X = df.drop(columns=["attack_label"])
    y = df["attack_label"]
    
    return X.values, y.values, X.columns
