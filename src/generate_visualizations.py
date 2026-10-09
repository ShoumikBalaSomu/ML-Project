"""
Visualization generator for Credit Card Fraud Detection.
Produces high-resolution figures highlighting data characteristics,
Scikit-Learn 8.3 preprocessing transformations, and model evaluation.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve

from preprocessing import (
    load_dataset,
    balance_dataset_undersample,
    get_scaler,
    get_power_transformer,
    build_preprocessing_pipeline
)
from model import get_models, evaluate_model


def generate_all_figures(data_path: str, output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    print(f"Loading data from {data_path}...")
    df = load_dataset(data_path)
    print("Dataset loaded:", df.shape)

    # -------------------------------------------------------------
    # Figure 1: Class Distribution (Imbalance)
    # -------------------------------------------------------------
    print("Generating Figure 1: Class Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    counts = df['Class'].value_counts()
    percentages = df['Class'].value_counts(normalize=True) * 100
    
    # Linear scale
    sns.barplot(x=counts.index, y=counts.values, palette=["#2ecc71", "#e74c3c"], ax=axes[0])
    axes[0].set_title("Class Distribution (Linear Scale)")
    axes[0].set_xlabel("Transaction Class (0 = Legit, 1 = Fraud)")
    axes[0].set_ylabel("Number of Transactions")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 2000, f"{v:,}\n({percentages[i]:.3f}%)", ha='center', fontweight='bold')
    
    # Log scale
    sns.barplot(x=counts.index, y=counts.values, palette=["#2ecc71", "#e74c3c"], ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set_title("Class Distribution (Logarithmic Scale)")
    axes[1].set_xlabel("Transaction Class (0 = Legit, 1 = Fraud)")
    axes[1].set_ylabel("Log(Transactions Count)")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v * 1.2, f"{v:,}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Scikit-Learn 8.3 Preprocessing - Scaling and Transformation Comparison
    # -------------------------------------------------------------
    print("Generating Figure 2: Preprocessing Transformations Comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Raw Amount (subset for clean plotting)
    amt = df['Amount'].values.reshape(-1, 1)
    
    # 1. Raw Amount
    sns.histplot(amt, bins=50, kde=True, color="#34495e", ax=axes[0, 0])
    axes[0, 0].set_title("Raw Amount (Extreme Right Skewness, Max=$25.6K)")
    axes[0, 0].set_xlabel("Amount ($)")
    axes[0, 0].set_xlim(0, 1000)

    # 2. StandardScaler (8.3.1)
    std_scaler = get_scaler("standard")
    amt_std = std_scaler.fit_transform(amt)
    sns.histplot(amt_std, bins=50, kde=True, color="#3498db", ax=axes[0, 1])
    axes[0, 1].set_title("Scikit-Learn 8.3.1: StandardScaler (Mean=0, Std=1)")
    axes[0, 1].set_xlabel("Standardized Amount")
    axes[0, 1].set_xlim(-1, 5)

    # 3. RobustScaler (8.3.1)
    rob_scaler = get_scaler("robust")
    amt_rob = rob_scaler.fit_transform(amt)
    sns.histplot(amt_rob, bins=50, kde=True, color="#e67e22", ax=axes[1, 0])
    axes[1, 0].set_title("Scikit-Learn 8.3.1: RobustScaler (Median & IQR, Outlier-Resistant)")
    axes[1, 0].set_xlabel("Robust Scaled Amount")
    axes[1, 0].set_xlim(-1, 10)

    # 4. PowerTransformer Yeo-Johnson (8.3.2)
    power_tx = get_power_transformer("yeo-johnson")
    # Fit on sample to speed up calculation
    amt_sample = df['Amount'].sample(n=20000, random_state=42).values.reshape(-1, 1)
    amt_power = power_tx.fit_transform(amt_sample)
    sns.histplot(amt_power, bins=50, kde=True, color="#9b59b6", ax=axes[1, 1])
    axes[1, 1].set_title("Scikit-Learn 8.3.2: PowerTransformer (Yeo-Johnson to Gaussian)")
    axes[1, 1].set_xlabel("Transformed Amount (Gaussian Normalized)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "amount_scaling_comparison.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: Time Distribution (Fraud vs Legit)
    # -------------------------------------------------------------
    print("Generating Figure 3: Transaction Time Distribution...")
    fig, ax = plt.subplots(figsize=(12, 5))
    df_plot = df.copy()
    df_plot['Hours'] = df_plot['Time'] / 3600.0  # Convert seconds to hours
    
    sns.histplot(df_plot[df_plot.Class == 0]['Hours'], bins=48, color="#2ecc71", alpha=0.5, stat="density", label="Legitimate", ax=ax)
    sns.histplot(df_plot[df_plot.Class == 1]['Hours'], bins=48, color="#e74c3c", alpha=0.6, stat="density", label="Fraudulent", ax=ax)
    ax.set_title("Transaction Time Distribution Over 48-Hour Window")
    ax.set_xlabel("Time Elapsed (Hours)")
    ax.set_ylabel("Transaction Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transaction_time_distribution.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Figure 4: Balanced Dataset Correlation Heatmap
    # -------------------------------------------------------------
    print("Generating Figure 4: Correlation Matrix...")
    balanced_df = balance_dataset_undersample(df, random_state=42)
    plt.figure(figsize=(16, 12))
    corr = balanced_df.corr()
    sns.heatmap(corr, cmap="coolwarm_r", annot=False, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix (Balanced Dataset: 492 Legit vs 492 Fraud)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Model Training on Balanced Dataset with Preprocessing Pipeline
    # -------------------------------------------------------------
    print("Training models on preprocessed balanced dataset...")
    X = balanced_df.drop("Class", axis=1)
    y = balanced_df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scikit-learn preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(scale_method="robust")
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    models = get_models(random_state=42)
    eval_results = {}
    for name, model in models.items():
        print(f"  Evaluating {name}...")
        res = evaluate_model(model, X_train_prep, y_train, X_test_prep, y_test)
        eval_results[name] = res

    # -------------------------------------------------------------
    # Figure 5: Confusion Matrices
    # -------------------------------------------------------------
    print("Generating Figure 5: Confusion Matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for idx, (name, res) in enumerate(eval_results.items()):
        cm = res["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=["Pred Legit", "Pred Fraud"],
                    yticklabels=["True Legit", "True Fraud"])
        axes[idx].set_title(f"{name}\nAcc: {res['test_accuracy']:.4f} | Recall: {res['recall']:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Figure 6: ROC and Precision-Recall Curves
    # -------------------------------------------------------------
    print("Generating Figure 6: ROC & PR Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, res in eval_results.items():
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, res["y_test_proba"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.4f})", linewidth=2)

        # Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_test, res["y_test_proba"])
        axes[1].plot(rec, prec, label=f"{name} (AP = {res['average_precision']:.4f})", linewidth=2)

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.6)
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_pr_curves.png"), dpi=300)
    plt.close()

    print("All figures successfully generated in:", output_dir)
    return eval_results


if __name__ == "__main__":
    import sys
    data_path = "/home/krishna/Downloads/creditcard.csv"
    if not os.path.exists(data_path):
        data_path = "/home/krishna/ML-Project/data/creditcard_balanced.csv"
    generate_all_figures(data_path, "/home/krishna/ML-Project/figures")
