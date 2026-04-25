# Project Insights & Results

## 1. Data Pipeline
- **Source:** Dataset loaded from `../Dataset/Advertising Budget and Sales.csv`.
- **Cleaning:** Missing values filled with column mean, duplicates removed.
- **Scaling:** Features scaled using StandardScaler.
- **Output:** Cleaned data saved to `data/cleaned_advertising.csv`.

## 2. EDA
- **Correlation Heatmap:** See `reports/figures/correlation_heatmap.png`.
- **Pairplot:** See `reports/figures/pairplot.png`.
- **Feature vs Sales:** See `reports/figures/TV_vs_Sales.png`, `Radio_vs_Sales.png`, `Newspaper_vs_Sales.png`.
- **Key Insights:**
  - TV budget has the strongest correlation with Sales.
  - Radio has moderate correlation.
  - Newspaper has weak correlation.
- **Details:** See `reports/eda_insights.md`.

## 3. Model Training & Tuning
- **Models:** Linear Regression, Random Forest, Gradient Boosting.
- **Best Model:** See `reports/model_metrics.csv` and `reports/tuning_results.txt`.
- **Feature Importance:** See `reports/feature_importance.csv`.
- **Saved Model:** `models/<BestModel>_tuned_model.pkl`.

## 4. Evaluation
- **Metrics:** See `reports/eval_metrics.txt`.
- **Actual vs Predicted Plot:** `reports/figures/actual_vs_predicted.png`.

## 5. Prediction System
- **Reusable function:** `src/predict.py`.
- **Inputs:** TV, Radio, Newspaper budgets.
- **Output:** Predicted Sales.

## 6. Dashboard
- **Streamlit app:** `app/dashboard.py`.
- **Features:** Interactive prediction, budget visualization, channel recommendation, model insights.

---

# Recommendations
- Focus more on TV and Radio for higher ROI.
- Use the dashboard to experiment with budget allocation.

---

# File Connectivity & Flow
- **Dataset:** Place raw CSV in `Dataset/`.
- **Pipeline:** `src/data_pipeline.py` loads from `../Dataset/Advertising Budget and Sales.csv`, outputs to `data/cleaned_advertising.csv`.
- **EDA:** `src/eda.py` uses cleaned data, saves plots/insights to `reports/`.
- **Modeling:** `src/train_models.py` and `src/model_tuning.py` use cleaned data, save models to `models/` and metrics to `reports/`.
- **Evaluation:** `src/evaluate.py` uses tuned model and cleaned data, saves results to `reports/`.
- **Prediction:** `src/predict.py` loads tuned model and scaler, used by dashboard and notebooks.
- **Dashboard:** `app/dashboard.py` connects to all outputs for live prediction and recommendations.

---

# How to Run
1. Place your dataset in `Dataset/Advertising Budget and Sales.csv`.
2. Run each notebook in `notebooks/` in order, or run scripts in `src/`.
3. Launch the dashboard with `streamlit run app/dashboard.py`.

---

# See README.md for full instructions.