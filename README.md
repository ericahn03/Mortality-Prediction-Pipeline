# Mortality Rate Prediction with Machine Learning

This project applies various machine learning models to predict **age-specific mortality rates** using real-world global health data from the **Institute for Health Metrics and Evaluation (IHME)**. The models include both baseline and ensemble methods, evaluated using appropriate metrics and cross-validation techniques.

---

## Objective

To compare and evaluate the performance of **baseline regressors** and **ensemble learning models** for predicting mortality rates per 100,000 people, based on age, country, sex, and year.

---

## Dataset

| File Name               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `IHME_GBD_countrydata.csv` | Raw dataset from IHME including demographic and death statistics           |
| `preprocessed_data.csv`    | Cleaned version with numeric conversions, encoded features, and log columns |
| `model_results.csv`        | Model performance results (MAE, RMSE, R², Runtime, Eval method) - Baseline |
| `ensemble_results.csv`     | Performance metrics for ensemble models                                   |

---

## Python Scripts

| Script Name             | Functionality                                                                 |
|------------------------|--------------------------------------------------------------------------------|
| `preprocess_data.py`   | Cleans the raw data, maps age groups to numeric midpoints, encodes categoricals |
| `load_explore.py`      | Generates distributions, heatmaps, and scatterplots of feature relationships  |
| `train_models.py`      | Trains and evaluates:                                                          |
|                        | • Linear Regression (10-Fold CV)                                               |
|                        | • Decision Tree (10-Fold CV)                                                   |
|                        | • Support Vector Regression (80/20 Split, Polynomial Kernel)                   |
| `ensemble_models.py`   | Trains and evaluates ensemble models using 10-Fold CV:                         |
|                        | • Random Forest                                                                |
|                        | • Bagging (Decision Trees)                                                     |
|                        | • Stacking (Tree + Linear → Ridge)                                             |
| `visualize_results.py` | Generates bar charts comparing performance and feature importance              |
| `run_all.py`           | Automates the entire pipeline from preprocessing to visualization              |

---

## 🧪 Evaluation Metrics

- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Squared Error  
- **R² Score**: Coefficient of Determination  
- **Runtime**: Total training time in seconds  
- **Evaluation**: `10-Fold CV` or `80/20 Split` as specified per model  

---

## Model Highlights

- SVR uses an **80/20 Split** to match Weka’s behavior and reduce training time.
- All other models use **10-Fold Cross-Validation** for better reliability.
- Ensemble methods tend to outperform baseline models on most metrics.
- Random Forest feature importance is used to explain predictive drivers.

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# To Run:

```bash
python3 run_all.py
```
This executes:
- Data preprocessing
- Data exploration
- Model training
- Ensemble training
- Results visualization
