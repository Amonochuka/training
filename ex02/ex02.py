#!/usr/bin/env python3
"""
Exercise 2: Accuracy Scikit-learn
Compute the accuracy using sklearn.metrics on y_true and y_pred
"""
from sklearn.metrics import accuracy_score

y_pred = [0, 1, 0, 1, 0, 1, 0]
y_true = [0, 0, 1, 1, 1, 1, 0]

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")