"""
data_pipeline.py
----------------
Data loading, cleaning, and preprocessing pipeline for Advertising Sales Prediction project.
"""
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load CSV data into a pandas DataFrame."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean the dataset: handle missing values, remove duplicates."""
    # Drop duplicates
    df = df.drop_duplicates()
    # Fill missing values with column mean (for numeric columns)
    for col in ['TV', 'Radio', 'Newspaper', 'Sales']:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    return df

def scale_features(df, feature_cols):
    """Scale features using StandardScaler. Returns scaled DataFrame and scaler object."""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled, scaler

def save_cleaned_data(df, out_path):
    """Save cleaned DataFrame to CSV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    # Paths
    RAW_PATH = os.path.join("..", "..", "Advertising Budget and Sales.csv")
    CLEAN_PATH = os.path.join("..", "data", "cleaned_advertising.csv")
    # Load
    df = load_data(RAW_PATH)
    # Clean
    df_clean = clean_data(df)
    # Scale
    feature_cols = ['TV', 'Radio', 'Newspaper']
    df_scaled, scaler = scale_features(df_clean, feature_cols)
    # Save
    save_cleaned_data(df_scaled, CLEAN_PATH)
    print(f"Cleaned and scaled data saved to {CLEAN_PATH}")
