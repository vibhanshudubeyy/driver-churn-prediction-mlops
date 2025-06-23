import pandas as pd
from scipy.stats import ks_2samp

# List of features to check for drift
FEATURES_TO_MONITOR = ["earnings", "hours_worked", "tenure_months"]

def check_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    print("🔍 Data Drift Report:\n")
    for feature in FEATURES_TO_MONITOR:
        if feature not in reference_data.columns or feature not in current_data.columns:
            print(f"⚠️ Feature '{feature}' not found in one of the datasets.")
            continue

        # Drop NaNs to avoid false positives in KS test
        ref = reference_data[feature].dropna()
        curr = current_data[feature].dropna()

        # Kolmogorov-Smirnov test
        stat, p_value = ks_2samp(ref, curr)

        if p_value < 0.05:
            print(f"🚨 Drift detected in '{feature}' (p = {p_value:.4f})")
        else:
            print(f"✅ No drift in '{feature}' (p = {p_value:.4f})")

if __name__ == "__main__":
    reference_data = pd.read_csv("data/driver_data.csv")
    current_data = pd.read_csv("data/driver_data.csv")  # Replace with new data in production

    check_drift(reference_data, current_data)
