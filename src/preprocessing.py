"""
Preprocessing module for Credit Card Fraud Detection.
Adheres strictly to Scikit-Learn 8.3 Documentation:
https://scikit-learn.org/stable/modules/preprocessing.html

Covers:
- 8.3.1 Standardization and Scaling (StandardScaler, RobustScaler, MinMaxScaler)
- 8.3.2 Non-linear Transformation (PowerTransformer - Yeo-Johnson, QuantileTransformer)
- 8.3.5 Discretization & Binarization (KBinsDiscretizer, Binarizer)
- 8.3.6 Imputation of Missing Values (SimpleImputer)
- Comprehensive ColumnTransformer & Pipeline composition to avoid data leakage
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    KBinsDiscretizer,
    Binarizer,
    FunctionTransformer
)


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load transaction dataset from CSV file."""
    df = pd.read_csv(file_path)
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Compute key diagnostic information about the dataset.
    Returns summary stats, missing values, and class imbalance ratio.
    """
    total_samples = len(df)
    class_counts = df['Class'].value_counts().to_dict()
    legit_count = class_counts.get(0, 0)
    fraud_count = class_counts.get(1, 0)
    fraud_percentage = (fraud_count / total_samples) * 100 if total_samples > 0 else 0.0

    missing_values = df.isnull().sum().to_dict()
    total_missing = sum(missing_values.values())

    return {
        "total_samples": total_samples,
        "features_count": df.shape[1],
        "legit_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_percentage": fraud_percentage,
        "total_missing_values": total_missing,
        "columns": list(df.columns)
    }


def balance_dataset_undersample(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Under-sampling method following Kaggle Project tutorial:
    'Detect Credit Card Frauds with python || Kaggle Projects 2023 || Easily Explained'

    Extracts all fraudulent transactions (n=492) and samples an equal number
    of legitimate transactions (n=492) to build a 50/50 balanced dataset.
    """
    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]
    
    n_fraud = len(fraud)
    legit_sample = legit.sample(n=n_fraud, random_state=random_state)
    
    balanced_df = pd.concat([legit_sample, fraud], axis=0)
    # Shuffle the dataset thoroughly
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced_df


def get_scaler(method: str = "robust"):
    """
    Scikit-Learn 8.3.1: Standardization and Scaling.
    Options:
    - 'standard': StandardScaler (z = (x - u) / s)
    - 'robust': RobustScaler (median and IQR, resistant to financial outliers)
    - 'minmax': MinMaxScaler (scales features to range [0, 1])
    """
    method = method.lower()
    if method == "robust":
        return RobustScaler()
    elif method == "standard":
        return StandardScaler()
    elif method == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method '{method}'. Choose 'robust', 'standard', or 'minmax'.")


def get_power_transformer(method: str = "yeo-johnson"):
    """
    Scikit-Learn 8.3.2: Non-linear transformation.
    PowerTransformer maps features towards a Gaussian distribution to stabilize variance
    and minimize skewness (essential for right-skewed transaction Amounts).
    """
    return PowerTransformer(method=method, standardize=True)


def get_quantile_transformer(output_distribution: str = "normal", random_state: int = 42):
    """
    Scikit-Learn 8.3.2: QuantileTransformer.
    Transforms features using quantiles information to follow uniform or normal distribution.
    """
    return QuantileTransformer(output_distribution=output_distribution, random_state=random_state)


def get_discretizer(n_bins: int = 5, encode: str = "ordinal", strategy: str = "uniform"):
    """
    Scikit-Learn 8.3.5: Discretization (KBinsDiscretizer).
    Partitions continuous features (e.g., Time) into discrete intervals.
    """
    return KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)


def build_preprocessing_pipeline(
    scale_method: str = "robust",
    apply_power_transform: bool = False,
    include_discretization: bool = False
) -> ColumnTransformer:
    """
    Build a comprehensive Scikit-Learn ColumnTransformer pipeline.
    Combines:
    - Imputation (8.3.6): SimpleImputer(strategy='median')
    - Scaling (8.3.1): RobustScaler or StandardScaler for Time and Amount
    - Non-linear Transformation (8.3.2): Optional PowerTransformer for Amount
    - PCA features (V1-V28): Standardized or passed through
    
    This ensures all fit operations occur strictly on training data,
    completely preventing data leakage into test sets.
    """
    pca_features = [f"V{i}" for i in range(1, 29)]
    
    # Amount pipeline
    amount_steps = [("imputer", SimpleImputer(strategy="median"))]
    if apply_power_transform:
        amount_steps.append(("power_transform", PowerTransformer(method="yeo-johnson")))
    else:
        amount_steps.append(("scaler", get_scaler(scale_method)))
    amount_pipeline = Pipeline(amount_steps)

    # Time pipeline
    time_steps = [("imputer", SimpleImputer(strategy="median"))]
    if include_discretization:
        time_steps.append(("discretizer", KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="uniform")))
    else:
        time_steps.append(("scaler", get_scaler(scale_method)))
    time_pipeline = Pipeline(time_steps)

    # PCA features pipeline (ensures zero mean, unit variance and handles potential NaNs)
    pca_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("amount", amount_pipeline, ["Amount"]),
            ("time", time_pipeline, ["Time"]),
            ("pca", pca_pipeline, pca_features)
        ],
        remainder="drop"
    )

    return preprocessor
