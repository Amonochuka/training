#!/usr/bin/env python3
"""
Exercise 1: MSE Scikit-learn
Compute the MSE using sklearn.metrics on y_true and y_pred
"""
from sklearn.metrics import mean_squared_error

y_true = [91, 51, 2.5, 2, -5]
y_pred = [90, 48, 2, 2, -4]

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse}")