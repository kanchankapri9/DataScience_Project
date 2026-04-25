
# Smart Advertising Sales Prediction & Budget Optimizer

A production-ready ML system for predicting sales and optimizing advertising budgets using advanced machine learning and interactive dashboards.

---

## Project Structure

```
ad_sales_project/
│
├── Dataset/                         # Place your raw CSV here
│   └── Advertising Budget and Sales.csv
├── data/                            # Cleaned data
├── notebooks/                       # Jupyter notebooks for each step
├── src/                             # Modular Python scripts
├── models/                          # Saved ML models
├── app/                             # Streamlit dashboard
├── reports/                         # EDA charts, metrics, insights
├── requirements.txt
└── README.md
```

---

## Workflow & File Connectivity

1. **Data Pipeline**
	- `src/data_pipeline.py`: Loads data from `Dataset/Advertising Budget and Sales.csv`, cleans, scales, and saves to `data/cleaned_advertising.csv`.
2. **EDA**
	- `src/eda.py`: Uses cleaned data, generates plots in `reports/figures/`, insights in `reports/eda_insights.md`.
3. **Model Training**
	- `src/train_models.py`: Trains models, saves metrics to `reports/model_metrics.csv`, best model to `models/`.
4. **Model Tuning**
	- `src/model_tuning.py`: Hyperparameter tuning, saves tuned model and feature importance to `reports/`.
5. **Evaluation**
	- `src/evaluate.py`: Evaluates tuned model, saves metrics and plots to `reports/`.
6. **Prediction System**
	- `src/predict.py`: Reusable prediction function, used by dashboard and notebooks.
7. **Dashboard**
	- `app/dashboard.py`: Streamlit app for interactive prediction and recommendations.

---

## How to Run

1. **Install dependencies:**
	```
	pip install -r requirements.txt
	```
2. **Place your dataset:**
	- Put `Advertising Budget and Sales.csv` in the `Dataset/` folder.
3. **Run the pipeline:**
	- Use the notebooks in `notebooks/` or run scripts in `src/` in order:
	  - `data_pipeline.py` → `eda.py` → `train_models.py` → `model_tuning.py` → `evaluate.py`
4. **Test prediction:**
	- Use `notebooks/run_predict.ipynb` or `src/predict.py`.
5. **Launch dashboard:**
	```
	cd app
	streamlit run dashboard.py
	```

---

## Documentation

- **Each script is modular and documented.**
- **Notebooks** show step-by-step execution and results.
- **`reports/`** contains all generated plots, metrics, and insights.
- **`project_insights.md`** summarizes findings and recommendations.

---

## File Explanations

- **src/data_pipeline.py**: Data cleaning and scaling.
- **src/eda.py**: EDA, plots, and insights.
- **src/train_models.py**: Model training and selection.
- **src/model_tuning.py**: Hyperparameter tuning and feature importance.
- **src/evaluate.py**: Model evaluation and result visualization.
- **src/predict.py**: Reusable prediction function.
- **app/dashboard.py**: Streamlit dashboard for interactive use.
- **notebooks/**: Run and test each step interactively.
- **reports/**: All outputs, metrics, and insights.

---

## Author & License

- Built by a senior ML engineer.
- MIT License.