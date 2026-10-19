#!/usr/bin/env python3
"""
Exercise 6: Grid Search
Hyperparameter tuning using GridSearchCV on Random Forest
"""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load California Housing dataset
housing = fetch_california_housing()
X, y = housing['data'], housing['target']

# Create pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=43))
])

# Define parameter grid for GridSearchCV
param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [10, 20, None],
    'rf__min_samples_leaf': [1, 2, 4]
}

# Custom CV split as specified
cv_split = [(np.arange(18576), np.arange(18576, 20640))]

# Grid Search
gs = GridSearchCV(pipeline, param_grid, cv=cv_split, n_jobs=-1, scoring='r2', verbose=1)
gs.fit(X, y)

# Function to select best model
def select_model_verbose(gs):
    trained_model = gs.best_estimator_
    best_params = gs.best_params_
    best_score = gs.best_score_
    return trained_model, best_params, best_score

trained_model, best_params, best_score = select_model_verbose(gs)

print("\n=== Grid Search Results ===")
print(f"Best Parameters: {best_params}")
print(f"Best CV Score (R2): {best_score:.4f}")

# Predict on new point
new_point = np.array([[3.2031, 52., 5.47761194, 1.07960199, 910., 2.26368159, 37.85, -122.26]])
prediction = trained_model.predict(new_point)
print(f"\nPrediction on new point: {prediction[0]:.4f}")

# Also show train/test metrics for the best model
# Since GridSearchCV used all data with custom CV, let's evaluate on the same split
X_train, X_test = X[:18576], X[18576:]
y_train, y_test = y[:18576], y[18576:]

y_train_pred = trained_model.predict(X_train)
y_test_pred = trained_model.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)

r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print("\n=== Best Model Evaluation ===")
print(f"Train - R2: {r2_train:.4f}, MSE: {mse_train:.4f}, MAE: {mae_train:.4f}")
print(f"Test  - R2: {r2_test:.4f}, MSE: {mse_test:.4f}, MAE: {mae_test:.4f}")