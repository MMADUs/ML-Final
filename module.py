import pickle
import pandas as pd

MODEL_PATH = "stacked_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


def feature_engineering(df):
    df = df.copy()

    df["n_p_ratio"] = df["nitrogen"] / (df["phosphorus"] + 1e-6)
    df["n_k_ratio"] = df["nitrogen"] / (df["potassium"] + 1e-6)
    df["p_k_ratio"] = df["phosphorus"] / (df["potassium"] + 1e-6)

    df["npk_total"] = df["nitrogen"] + df["phosphorus"] + df["potassium"]

    df["humidity_rain_ratio"] = df["humidity"] / (df["rainfall"] + 1e-6)

    df["ph_type"] = pd.cut(
        df["ph"], bins=[0, 6.0, 7.5, 14], labels=["acidic", "neutral", "alkaline"]
    )

    df["temp_zone"] = pd.cut(
        df["temperature"], bins=[0, 15, 25, 50], labels=["cool", "moderate", "hot"]
    )

    return df


def _build_df(n, p, k, temperature, humidity, ph, rainfall):
    return pd.DataFrame(
        [
            {
                "nitrogen": n,
                "phosphorus": p,
                "potassium": k,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            }
        ]
    )


def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    df = feature_engineering(_build_df(n, p, k, temperature, humidity, ph, rainfall))
    prediction = model.predict(df)

    with open(LABEL_ENCODER_PATH, "rb") as file:
        le = pickle.load(file)

    return le.inverse_transform(prediction)[0]


def predict_crop_proba(n, p, k, temperature, humidity, ph, rainfall):
    """Returns (proba_dict, engineered_features_dict)."""
    df_raw = _build_df(n, p, k, temperature, humidity, ph, rainfall)
    df_eng = feature_engineering(df_raw)

    with open(LABEL_ENCODER_PATH, "rb") as file:
        le = pickle.load(file)

    proba = model.predict_proba(df_eng)[0]
    proba_dict = {
        str(le.inverse_transform([i])[0]): float(p) for i, p in enumerate(proba)
    }

    row = df_eng.iloc[0]
    engineered = {
        "N/P Ratio": round(float(row["n_p_ratio"]), 4),
        "N/K Ratio": round(float(row["n_k_ratio"]), 4),
        "P/K Ratio": round(float(row["p_k_ratio"]), 4),
        "NPK Total": round(float(row["npk_total"]), 2),
        "Humidity/Rain Ratio": round(float(row["humidity_rain_ratio"]), 4),
        "pH Type": str(row["ph_type"]),
        "Temperature Zone": str(row["temp_zone"]),
    }

    return proba_dict, engineered
