#!/usr/bin/env python3
"""
Exercise 4: Classification
Evaluate a machine learning model using classification metrics
"""
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             roc_curve)
import matplotlib.pyplot as plt
import numpy as np

# Preliminary
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=43)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train_scaled, y_train)

# Predict on train and test sets
y_train_pred = classifier.predict(X_train_scaled)
y_test_pred = classifier.predict(X_test_scaled)

# Get probabilities for AUC (AUC requires probabilities, not classes)
y_train_prob = classifier.predict_proba(X_train_scaled)[:, 1]
y_test_prob = classifier.predict_proba(X_test_scaled)[:, 1]

# Compute metrics on train set
f1_train = f1_score(y_train, y_train_pred)
acc_train = accuracy_score(y_train, y_train_pred)
prec_train = precision_score(y_train, y_train_pred)
rec_train = recall_score(y_train, y_train_pred)
roc_auc_train = roc_auc_score(y_train, y_train_prob)

# Compute metrics on test set
f1_test = f1_score(y_test, y_test_pred)
acc_test = accuracy_score(y_test, y_test_pred)
prec_test = precision_score(y_test, y_test_pred)
rec_test = recall_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_prob)

print("=== Train Set Metrics ===")
print(f"F1 Score: {f1_train:.4f}")
print(f"Accuracy: {acc_train:.4f}")
print(f"Precision: {prec_train:.4f}")
print(f"Recall: {rec_train:.4f}")
print(f"ROC AUC: {roc_auc_train:.4f}")

print("\n=== Test Set Metrics ===")
print(f"F1 Score: {f1_test:.4f}")
print(f"Accuracy: {acc_test:.4f}")
print(f"Precision: {prec_test:.4f}")
print(f"Recall: {rec_test:.4f}")
print(f"ROC AUC: {roc_auc_test:.4f}")

# Confusion matrix on test set
cm = confusion_matrix(y_test, y_test_pred)
print("\n=== Confusion Matrix (Test Set) ===")
print(cm)

# Plot ROC curve for test set
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc_test:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Breast Cancer Classification (Test Set)')
plt.legend()
plt.grid(True)
plt.savefig('w2_day4_ex4_q3.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nROC curve saved as w2_day4_ex4_q3.png")