import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sqlalchemy import create_engine, text
import mlflow
import mlflow.sklearn
import joblib
import schedule
import time

# MySQL connection
engine = create_engine("mysql+mysqlconnector://root:root@localhost:3306/driver_churn_db")

def extract_data():
    data = pd.read_csv("data/driver_data.csv")
    return data

def transform_data(data):

    if data["hours_worked"].notnull().any():
        data["hours_worked"] = data["hours_worked"].fillna(data["hours_worked"].median())
    else:
        data["hours_worked"] = data["hours_worked"].fillna(0)

    data["city"] = data["city"].str.title()
    data["earnings_per_delivery"] = data["earnings"] / data["deliveries_completed"]
    data = pd.get_dummies(data, columns=["city"], drop_first=False)
    return data

def train_model(data):
    X = data.drop(["driver_id", "churn"], axis=1)
    y = data["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "churn_model")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_size", 0.2)
        mlflow.set_tag("model", "RandomForestClassifier")
        joblib.dump(model, "models/churn_model.pkl")
        return model, f1

def load_predictions(data, model):
    X = data.drop(["driver_id", "churn"], axis=1)
    predictions = model.predict(X)
    data["churn_prediction"] = predictions
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS churn_predictions (
                driver_id INT PRIMARY KEY,
                churn_prediction INT
                )
            """))

    data[["driver_id", "churn_prediction"]].to_sql("churn_predictions", engine, if_exists="replace", index=False)

def run_pipeline():
    data = extract_data()
    transformed_data = transform_data(data)
    model, f1 = train_model(transformed_data)
    load_predictions(transformed_data, model)
    print(f"Pipeline executed. F1-score: {f1}")

# Schedule pipeline to run daily
schedule.every(1).days.do(run_pipeline)

if __name__ == "__main__":
    run_pipeline()
    while True:
        schedule.run_pending()
        time.sleep(60)