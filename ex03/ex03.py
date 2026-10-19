#!/usr/bin/env python3
"""
Exercise 3: Regression
Evaluate a machine learning model using regression metrics (R2, MSE, MAE)
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
                                                    random_state=13)

# pipeline
pipeline = [('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())]
pipe = Pipeline(pipeline)

# fit
pipe.fit(X_train, y_train)

# Predict on train and test sets
y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)

# Compute metrics on train set
r2_train = r2_score(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)

# Compute metrics on test set
r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print("=== Train Set Metrics ===")
print(f"R2 Score: {r2_train:.4f}")
print(f"MSE: {mse_train:.4f}")
print(f"MAE: {mae_train:.4f}")

print("\n=== Test Set Metrics ===")
print(f"R2 Score: {r2_test:.4f}")
print(f"MSE: {mse_test:.4f}")
print(f"MAE: {mae_test:.4f}")