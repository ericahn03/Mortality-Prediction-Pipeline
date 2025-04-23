# Mortality Analysis Models

*This project explores different machine learning models to predict age-specific death rates using publicly available IHME mortality data. It includes preprocessing, training baseline and ensemble models, and visualizing performance.*

**Objective:**  
To evaluate and compare the performance of base regression models and ensemble methods on real-world mortality data using metrics such as MAE, RMSE, R² Score, and training time.

---

## How to Run the Pipeline

*Ensure Python 3.8+ and the following packages are installed:*
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

*Use this command to run all steps in sequence:*
```bash
python3 run_all.py
```
This will:
1. Preprocess the raw IHME dataset
2. Run exploratory data analysis (EDA)
3. Train baseline models (Linear Regression, Decision Tree, SVR)
4. Train ensemble models (Random Forest, Bagging, Stacking)
5. Generate performance visualizations

---

## Data Files: 

File Name | Description
IHME_GBD_countrydata.csv | Raw dataset from the Institute for Health Metrics and Evaluation (IHME)
preprocessed_data.csv | Cleaned and enriched dataset including year, age midpoints, log-deaths
model_results.csv | Model evaluation output with metrics (MAE, RMSE, R², runtime)

---

## Python Scripts:

Script Name | Functionality
preprocess_data.py | Cleans the raw data, encodes categorical values, transforms age & deaths
load_explore.py | Performs data visualization and correlation heatmaps
train_models.py | Trains Linear Regression, Decision Tree, and SVR models
ensemble_models.py | Trains Random Forest, Bagging, and Stacking models with Ridge meta-model
visualize_results.py | Creates comparative bar plots for all model metrics
run_all.py | Executes the full pipeline end-to-end in correct order

---

## Evaluation Metrics:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² Score — Variance explained by the model
- Run Time (s) — Time taken to train each model

---

## Notes:

- All scripts assume you're in the root directory of the project.
- Run preprocess_data.py before any model training to ensure required columns exist.
- The "Age Group" column is transformed into a numeric "Age (midpoint)" for modeling.
- Any NaNs introduced via mapping or cleaning are dropped to avoid breaking models like SVR.
- Random Forest and Bagging consistently yielded the most robust results.
- SVR is slow and less scalable — use with care or tune hyperparameters for efficiency.