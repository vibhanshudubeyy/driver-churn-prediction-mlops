from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("models/churn_model.pkl")

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    df["earnings_per_delivery"] = df["earnings"] / df["deliveries_completed"]
    df = pd.get_dummies(df, columns=["city"], drop_first=True)
    training_columns = model.feature_names_in_
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[training_columns]
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0].tolist()
    return {"churn_prediction": int(prediction[0]), "probability": probability}