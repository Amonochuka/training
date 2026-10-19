#!/usr/bin/env python3
"""
Exercise 5: Machine Learning Models
Compare multiple ML algorithms: Linear Regression, SVM, Decision Tree, Random Forest, Gradient Boosting
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# imports
housing = fetch_california_housing()
X, y = housing['data'], housing['target']

# split data train test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    shuffle=True,
                                                    random_state=43)

# Base pipeline steps (imputer and scaler)
base_steps = [('imputer', SimpleImputer(strategy='median')),
              ('scaler', StandardScaler())]

# Create 5 pipelines with different models
models = {
    'Linear Regression': LinearRegression(),
    'SVM': SVR(),
    'Decision Tree': DecisionTreeRegressor(random_state=43),
    'Random Forest': RandomForestRegressor(random_state=43),
    'Gradient Boosting': GradientBoostingRegressor(random_state=43)
}

for name, model in models.items():
    pipeline = Pipeline(base_steps + [(name.lower().replace(' ', '_'), model)])
    pipeline.fit(X_train, y_train)
    
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n=== {name} ===")
    print(f"Train - R2: {r2_train:.4f}, MSE: {mse_train:.4f}, MAE: {mae_train:.4f}")
    print(f"Test  - R2: {r2_test:.4f}, MSE: {mse_test:.4f}, MAE: {mae_test:.4f}")