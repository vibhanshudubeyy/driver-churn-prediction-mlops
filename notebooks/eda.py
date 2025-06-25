import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import boto3
from io import StringIO


S3_BUCKET = "churn-data-vibhanshu"
S3_KEY = "driver_data.csv"

def extract_data():
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        csv_content = obj["Body"].read().decode("utf-8")
        data = pd.read_csv(StringIO(csv_content))
        return data
    except Exception as e:
        print(f"Error reading from S3: {e}")
        raise

# Load the dataset
data = extract_data()

# Basic Info
print("🔎 Dataset Info:")
print(data.info())

# Shape of the dataset
print(f"\n📏 Dataset Shape: {data.shape[0]} rows, {data.shape[1]} columns")

# First few rows
print("\n👀 First 5 Rows:")
print(data.head())

# Column-wise missing values and percentage
missing = data.isnull().sum()
missing_percent = (missing / len(data)) * 100
missing_df = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_percent})
print("\n⚠️ Missing Values Summary:")
print(missing_df[missing_df['Missing Values'] > 0])

# Summary statistics
print("\n📊 Summary Statistics (Numerical Features):")
print(data.describe())

# Unique values per column
print("\n🔢 Unique Values per Column:")
print(data.nunique())

# Value counts for categorical columns
print("\n📈 Categorical Columns Distribution:")
categorical_cols = data.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"\n🔹 {col} Distribution:")
    print(data[col].value_counts())

# Churn distribution (target variable)
if "churn" in data.columns:
    print("\n📌 Churn Distribution:")
    print(data["churn"].value_counts(normalize=True))
    sns.countplot(x='churn', data=data)
    plt.title("Churn Distribution")
    plt.show()

# Correlation matrix (only for numeric features)
print("\n🔗 Correlation Matrix:")
corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Skewness of numerical features
print("\n📐 Skewness of Numerical Features:")
numeric_cols = data.select_dtypes(include=np.number).columns
print(data[numeric_cols].skew())

# Outlier detection (IQR method preview)
print("\n🚨 Outlier Preview (using IQR method):")
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
    if outliers > 0:
        print(f"{col}: {outliers} potential outliers")

