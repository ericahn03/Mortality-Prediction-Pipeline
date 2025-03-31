# CS323 Machine Learning Project

*This project explores different machine learning models to predict age-specific death rates using publicly available IHME mortality data. It includes preprocessing, training baseline and ensemble models, and visualizing performance.*

*Objective: To evaluate and compare the performance of base regression models and ensemble methods on real-world mortality data using metrics such as MAE, RMSE, R² Score, and training time.*

---------------------------------------------------------------

## How to Run:

*Ensure Python 3 and required packages are installed:*
```python
pip install pandas numpy matplotlib seaborn scikit-learn
```
*Then, run the full pipeline:*
```python
python3 run_all.py
```
---------------------------------------------------------------

## Data:

- IHME_GBD_countrydata.csv - Raw mortality dataset
- preprocessed_data.csv - Cleaned and encoded dataset for modeling
- model_results.csv - Saved performance results from model training

## Python Scripts:

- pre_process.py - Cleans and encodes the data
- train_models.py - Trains base regressors (Linear, Decision Tree, SVR)
- ensemble_models.py - Trains ensemble regressors (Random Forest, Bagging, Stacking)
- visualize_results.py - Plots model performance comparison charts
- run_all.py - Automates the full pipeline

## Metrics Tracked:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Run Time (in seconds)

## Notes:

- SVR may run slower on large datasets
- Ensemble methods (especially Random Forest) generally performed best