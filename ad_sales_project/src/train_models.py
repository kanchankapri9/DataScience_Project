"""
train_models.py
---------------
Trains multiple regression models, evaluates them, and saves the best model.
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

def load_data(path):
    return pd.read_csv(path)

def split_features_target(df):
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    return X, y

def train_and_evaluate_models(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {'model': model, 'r2': r2, 'rmse': rmse}
    return results

def select_best_model(results):
    # Select by highest R2, then lowest RMSE
    best = max(results.items(), key=lambda x: (x[1]['r2'], -x[1]['rmse']))
    return best[0], best[1]

def save_model(model, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)

def save_metrics(results, out_path):
    lines = ['Model, R2 Score, RMSE']
    for name, res in results.items():
        lines.append(f"{name}, {res['r2']:.4f}, {res['rmse']:.4f}")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

def run_training():
    data_path = os.path.join('..', 'data', 'cleaned_advertising.csv')
    model_dir = os.path.join('..', 'models')
    metrics_path = os.path.join('..', 'reports', 'model_metrics.csv')
    df = load_data(data_path)
    X, y = split_features_target(df)
    results = train_and_evaluate_models(X, y)
    save_metrics(results, metrics_path)
    best_name, best = select_best_model(results)
    model_path = os.path.join(model_dir, f'{best_name}_best_model.pkl')
    save_model(best['model'], model_path)
    print(f"Best model: {best_name} (R2: {best['r2']:.4f}, RMSE: {best['rmse']:.4f})")
    print(f"Model saved to {model_path}")
    print(f"All metrics saved to {metrics_path}")

if __name__ == "__main__":
    run_training()
