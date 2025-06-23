from fastapi import FastAPI
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI()

# Resolve the absolute path to the model file
BASE_DIR = Path(__file__).resolve().parents[1]  # Go one level up to project root
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
model = joblib.load(MODEL_PATH)
from pydantic import BaseModel

class DriverData(BaseModel):
    driver_id: int
    deliveries_completed: int
    hours_worked: float
    earnings: float
    city: str
    tenure_months: int


@app.post("/predict")
async def predict(data: DriverData):
    df = pd.DataFrame([data.dict()])
    df["earnings_per_delivery"] = df["earnings"] / df["deliveries_completed"]
    df = pd.get_dummies(df, columns=["city"], drop_first=True)

    training_columns = model.feature_names_in_
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[training_columns]
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0].tolist()

    return {
        "churn_prediction": int(prediction[0]),
        "probability": probability
    }
