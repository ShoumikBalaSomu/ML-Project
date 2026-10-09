"""
Model training and evaluation module for Credit Card Fraud Detection.
Following Kaggle Project tutorial workflow:
'Detect Credit Card Frauds with python || Kaggle Projects 2023 || Easily Explained'

Supports Logistic Regression, Random Forest, and comprehensive evaluation metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def get_models(random_state: int = 42) -> Dict[str, Any]:
    """Returns baseline and advanced models for fraud detection."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver="liblinear"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
    }
    return models


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Fits the model on training data and computes comprehensive metrics
    on both train and test partitions to diagnose underfitting/overfitting.
    """
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = model.decision_function(X_test)

    # Accuracies (as in video tutorial)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Imbalanced classification metrics
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    avg_precision = average_precision_score(y_test, y_test_proba)
    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

    results = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_test_proba": y_test_proba,
        "y_test_pred": y_test_pred
    }
    return results
