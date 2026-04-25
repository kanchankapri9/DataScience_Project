"""
model_tuning.py
--------------
Performs hyperparameter tuning on the best model and analyzes feature importance.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib

def load_data(path):
    return pd.read_csv(path)

def split_features_target(df):
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    return X, y

def get_best_model_name(metrics_path):
    df = pd.read_csv(metrics_path)
    # Highest R2, then lowest RMSE
    df_sorted = df.sort_values(['R2 Score', 'RMSE'], ascending=[False, True])
    best_name = df_sorted.iloc[0]['Model']
    return best_name

def get_model_and_params(name):
    if name == 'RandomForest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
    elif name == 'GradientBoosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    elif name == 'LinearRegression':
        model = LinearRegression()
        param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}
    else:
        raise ValueError('Unknown model name')
    return model, param_grid

def tune_model(model, param_grid, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return best_model, grid.best_params_, r2, rmse

def save_model(model, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)

def save_tuning_results(params, r2, rmse, out_path):
    with open(out_path, 'w') as f:
        f.write(f"Best Params: {params}\nR2 Score: {r2:.4f}\nRMSE: {rmse:.4f}\n")

def save_feature_importance(model, feature_names, out_path):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        lines = ['Feature, Importance']
        for feat, imp in zip(feature_names, importances):
            lines.append(f"{feat}, {imp:.4f}")
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines))
    else:
        with open(out_path, 'w') as f:
            f.write('Feature importance not available for this model.')

def run_tuning():
    data_path = os.path.join('..', 'data', 'cleaned_advertising.csv')
    metrics_path = os.path.join('..', 'reports', 'model_metrics.csv')
    model_dir = os.path.join('..', 'models')
    tuning_path = os.path.join('..', 'reports', 'tuning_results.txt')
    feat_imp_path = os.path.join('..', 'reports', 'feature_importance.csv')
    df = load_data(data_path)
    X, y = split_features_target(df)
    best_name = get_best_model_name(metrics_path)
    model, param_grid = get_model_and_params(best_name)
    best_model, best_params, r2, rmse = tune_model(model, param_grid, X, y)
    model_path = os.path.join(model_dir, f'{best_name}_tuned_model.pkl')
    save_model(best_model, model_path)
    save_tuning_results(best_params, r2, rmse, tuning_path)
    save_feature_importance(best_model, X.columns, feat_imp_path)
    print(f"Tuned {best_name} model saved to {model_path}")
    print(f"Tuning results saved to {tuning_path}")
    print(f"Feature importance saved to {feat_imp_path}")

if __name__ == "__main__":
    run_tuning()
