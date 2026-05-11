# module.py

import pickle
import pandas as pd

MODEL_PATH = "stacked_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Load model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


def feature_engineering(df):
    # known ratios in agriculture
    df["n_p_ratio"] = df["nitrogen"] / (df["phosphorus"] + 1e-6)
    df["n_k_ratio"] = df["nitrogen"] / (df["potassium"] + 1e-6)
    df["p_k_ratio"] = df["phosphorus"] / (df["potassium"] + 1e-6)

    # total nutrient load
    df["npk_total"] = df["nitrogen"] + df["phosphorus"] + df["potassium"]

    # ratio to check whether humidity is caused by rainfall
    df["humidity_rain_ratio"] = df["humidity"] / (df["rainfall"] + 1e-6)

    # ph categorization
    df["ph_type"] = pd.cut(
        df["ph"], bins=[0, 6.0, 7.5, 14], labels=["acidic", "neutral", "alkaline"]
    )

    # temperature categorization
    df["temp_zone"] = pd.cut(
        df["temperature"], bins=[0, 15, 25, 50], labels=["cool", "moderate", "hot"]
    )

    return df


def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    # Create dataframe from streamlit input
    df = pd.DataFrame(
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

    # Apply feature engineering
    df = feature_engineering(df)

    # Predict
    prediction = model.predict(df)

    # Load label encoder
    with open(LABEL_ENCODER_PATH, "rb") as file:
        le = pickle.load(file)

    # Decode prediction
    decoded_prediction = le.inverse_transform(prediction)

    return decoded_prediction[0]
