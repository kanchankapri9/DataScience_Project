"""
evaluate.py
-----------
Evaluates the tuned model on the test set and generates evaluation plots.
"""
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def split_features_target(df):
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    return X, y

def load_best_tuned_model(models_dir):
    # Find the tuned model file
    for fname in os.listdir(models_dir):
        if fname.endswith('_tuned_model.pkl'):
            return joblib.load(os.path.join(models_dir, fname)), fname
    raise FileNotFoundError('No tuned model found.')

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return preds, r2, rmse

def plot_predictions(y_test, preds, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_eval_metrics(r2, rmse, out_path):
    with open(out_path, 'w') as f:
        f.write(f'R2 Score: {r2:.4f}\nRMSE: {rmse:.4f}\n')

def run_evaluation():
    data_path = os.path.join('..', 'data', 'cleaned_advertising.csv')
    models_dir = os.path.join('..', 'models')
    reports_dir = os.path.join('..', 'reports')
    fig_dir = os.path.join(reports_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    eval_metrics_path = os.path.join(reports_dir, 'eval_metrics.txt')
    eval_plot_path = os.path.join(fig_dir, 'actual_vs_predicted.png')
    df = load_data(data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, model_fname = load_best_tuned_model(models_dir)
    preds, r2, rmse = evaluate_model(model, X_test, y_test)
    save_eval_metrics(r2, rmse, eval_metrics_path)
    plot_predictions(y_test, preds, eval_plot_path)
    print(f"Evaluation complete for {model_fname}.")
    print(f"R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")
    print(f"Metrics saved to {eval_metrics_path}")
    print(f"Plot saved to {eval_plot_path}")

if __name__ == "__main__":
    run_evaluation()
