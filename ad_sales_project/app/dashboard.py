"""
dashboard.py
------------
Streamlit dashboard for Smart Advertising Sales Prediction & Budget Optimizer.
"""
import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_tuned_model(models_dir):
    for fname in os.listdir(models_dir):
        if fname.endswith('_tuned_model.pkl'):
            return joblib.load(os.path.join(models_dir, fname)), fname
    raise FileNotFoundError('No tuned model found.')

def load_scaler(data_path):
    df = pd.read_csv(data_path)
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

def recommend_channel(tv, radio, newspaper, feat_imp_path):
    # Suggest best channel based on feature importance
    if os.path.exists(feat_imp_path):
        df = pd.read_csv(feat_imp_path)
        if 'Importance' in df.columns:
            best_feat = df.sort_values('Importance', ascending=False).iloc[0]['Feature']
            return best_feat
    # Fallback: recommend channel with highest budget
    budgets = {'TV': tv, 'Radio': radio, 'Newspaper': newspaper}
    return max(budgets, key=budgets.get)

def main():
    st.title('Smart Advertising Sales Prediction & Budget Optimizer')
    st.write('Predict sales and optimize your advertising budget using machine learning.')

    st.sidebar.header('Set Advertising Budgets')
    tv = st.sidebar.slider('TV Budget', 0, 300, 100)
    radio = st.sidebar.slider('Radio Budget', 0, 60, 20)
    newspaper = st.sidebar.slider('Newspaper Budget', 0, 120, 30)

    if st.button('Predict Sales'):
        models_dir = os.path.join('..', 'models')
        data_path = os.path.join('..', 'data', 'cleaned_advertising.csv')
        feat_imp_path = os.path.join('..', 'reports', 'feature_importance.csv')
        pred, model_name = predict_sales(tv, radio, newspaper, models_dir, data_path)
        st.success(f'Predicted Sales: {pred:.2f} (using {model_name})')
        # Bar chart
        st.subheader('Budget Distribution')
        st.bar_chart({'Budget': [tv, radio, newspaper]}, x=['TV', 'Radio', 'Newspaper'])
        # Recommendation
        channel = recommend_channel(tv, radio, newspaper, feat_imp_path)
        st.info(f'Recommended channel for best ROI: **{channel}**')
        # Model insights
        if os.path.exists(feat_imp_path):
            st.subheader('Model Feature Importance')
            df = pd.read_csv(feat_imp_path)
            st.dataframe(df)

if __name__ == "__main__":
    main()
