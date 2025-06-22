import pandas as pd

data = pd.read_csv("data/driver_data.csv")
print("Dataset Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())
print("\nSummary Statistics:")
print(data.describe())
print("\nChurn Distribution:")
print(data["churn"].value_counts(normalize=True))