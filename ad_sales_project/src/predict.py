"""
predict.py
----------
Reusable prediction system for advertising sales.
"""
import os
import joblib
import numpy as np
import pandas as pd

def load_tuned_model(models_dir):
    for fname in os.listdir(models_dir):
        if fname.endswith('_tuned_model.pkl'):
            return joblib.load(os.path.join(models_dir, fname)), fname
    raise FileNotFoundError('No tuned model found.')

def load_scaler(data_path):
    # Fit scaler on the training data (for deployment, save scaler separately)
    df = pd.read_csv(data_path)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df[['TV', 'Radio', 'Newspaper']])
    return scaler

def predict_sales(tv, radio, newspaper, models_dir, data_path):
    model, model_name = load_tuned_model(models_dir)
    scaler = load_scaler(data_path)
    X = np.array([[tv, radio, newspaper]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return float(pred), model_name

if __name__ == "__main__":
    # Example usage
    models_dir = os.path.join('..', 'models')
    data_path = os.path.join('..', 'data', 'cleaned_advertising.csv')
    tv = float(input('Enter TV budget: '))
    radio = float(input('Enter Radio budget: '))
    newspaper = float(input('Enter Newspaper budget: '))
    pred, model_name = predict_sales(tv, radio, newspaper, models_dir, data_path)
    print(f"Predicted Sales: {pred:.2f} (using {model_name})")
