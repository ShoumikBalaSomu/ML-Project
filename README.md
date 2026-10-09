# CSE431: Machine Learning — Credit Card Fraud Detection & Data Preprocessing

| Notebook | Focus Area | Google Colab Link |
| :--- | :--- | :---: |
| **`Data_Preprocessing.ipynb`** | **Dedicated Data Preprocessing Script** (Scikit-Learn 8.3 Documentation) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShoumikBalaSomu/ML-Project/blob/main/Data_Preprocessing.ipynb) |
| **`Credit_Card_Fraud_Detection.ipynb`** | **End-to-End ML Pipeline** (EDA, Preprocessing, Modeling & Evaluation) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShoumikBalaSomu/ML-Project/blob/main/Credit_Card_Fraud_Detection.ipynb) |

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)
![Course](https://img.shields.io/badge/Course-CSE431--ML--Machine%20Learning-blueviolet.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Project Information
- **Course:** CSE431-ML-Machine Learning
- **Project Title:** Credit Card Fraud Detection
- **Student Name:** Shoumik Bala Somu
- **Email:** [shoumik.bala@gmail.com](mailto:shoumik.bala@gmail.com)
- **Repository:** [https://github.com/ShoumikBalaSomu/ML-Project](https://github.com/ShoumikBalaSomu/ML-Project)
- **Dedicated Preprocessing Notebook:** [`Data_Preprocessing.ipynb`](file:///home/krishna/ML-Project/Data_Preprocessing.ipynb)
- **End-to-End Project Notebook:** [`Credit_Card_Fraud_Detection.ipynb`](file:///home/krishna/ML-Project/Credit_Card_Fraud_Detection.ipynb)
- **Reference Video Tutorial:** [Detect Credit Card Frauds with python || Kaggle Projects 2023 || Easily Explained (YouTube)](https://www.youtube.com/watch?v=mggdxVcptxU)
- **Core Documentation Guide:** [8.3. Preprocessing data — scikit-learn documentation](https://scikit-learn.org/stable/modules/preprocessing.html)

---

## 📖 Table of Contents
1. [Project Overview & Problem Statement](#-project-overview--problem-statement)
2. [Dataset Description & Exploration](#-dataset-description--exploration)
3. [Scikit-Learn 8.3 Preprocessing Implementation](#-scikit-learn-83-preprocessing-implementation)
   - [8.3.1 Standardization and Scaling](#831-standardization-and-scaling)
   - [8.3.2 Non-linear Transformations](#832-non-linear-transformations)
   - [8.3.3 Normalization](#833-normalization)
   - [8.3.5 Discretization & Binarization](#835-discretization--binarization)
   - [8.3.6 Missing Value Imputation](#836-missing-value-imputation)
   - [8.3.7 Polynomial Features & 8.3.8 Custom Transformers](#837-polynomial-features--838-custom-transformers)
   - [Unified ColumnTransformer Pipeline](#unified-columntransformer-pipeline)
4. [Handling Extreme Class Imbalance](#-handling-extreme-class-imbalance)
5. [Model Training & Methodology](#-model-training--methodology)
6. [Experimental Results & Evaluation](#-experimental-results--evaluation)
7. [Repository Structure](#-repository-structure)
8. [How to Run (Google Colab & Local)](#-how-to-run)
9. [Conclusion & References](#-conclusion--references)

---

## 🎯 Project Overview & Problem Statement
Credit card fraud costs financial institutions and consumers billions of dollars annually. Building an automated fraud detection engine presents two major challenges:
1. **Extreme Class Imbalance:** Fraudulent transactions represent less than **0.173%** of all transactions ($492$ fraud cases out of $284,807$ total). A naive classifier that predicts all transactions as legitimate achieves $99.83\%$ accuracy while failing to detect any fraudulent events.
2. **Disparate Feature Scales & Heavy Outliers:** The dataset features $28$ PCA components ($V1$ through $V28$) which are centered around zero, whereas the raw **`Amount`** spans from $\$0.00$ to $\$25,691.16$ with extreme skewness, and **`Time`** spans $0$ to $172,792$ seconds ($48$ hours). Without principled preprocessing, models struggle with convergence and numerical stability.

In accordance with course requirements, this repository contains a **dedicated preprocessing notebook** [`Data_Preprocessing.ipynb`](file:///home/krishna/ML-Project/Data_Preprocessing.ipynb) that thoroughly implements the official **[Scikit-Learn 8.3 Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)** guide.

---

## 📊 Dataset Description & Exploration
The dataset contains transactions made by credit cards in September 2013 by European cardholders over a two-day period.

| Characteristic | Value |
| :--- | :--- |
| **Total Samples** | $284,807$ transactions |
| **Total Features** | $31$ columns ($30$ numerical features + $1$ target class) |
| **Target Column** | `Class` ($0$ = Legitimate, $1$ = Fraudulent) |
| **Legitimate Count** | $284,315$ ($99.827\%$) |
| **Fraudulent Count** | $492$ ($0.173\%$) |
| **Missing Values** | $0$ across all columns |
| **PCA Features** | $V1$ through $V28$ (transformed due to confidentiality) |
| **Unscaled Features**| `Time` (seconds elapsed) & `Amount` (transaction amount in USD) |

### Class Imbalance Visualization
![Class Distribution](figures/class_distribution.png)

### Transaction Time Analysis
![Transaction Time Distribution](figures/transaction_time_distribution.png)

---

## ⚙️ Scikit-Learn 8.3 Preprocessing Implementation
Following the specific instructions to implement **8.3. Preprocessing data — scikit-learn documentation**, our pipeline systematically covers all core subsections:

### 8.3.1 Standardization and Scaling
Gradient-based algorithms like Logistic Regression perform poorly when features operate at wildly different scales. We evaluated four scalers:
1. **`StandardScaler`**: Centers to $\mu=0$ and scales to $\sigma=1$:
   $$z = \frac{x - \mu}{\sigma}$$
   *Limitation:* Severe outliers in `Amount` (e.g. $\$25,691$) pull $\mu$ and inflate $\sigma$, compressing the majority of normal transactions into a tiny range.
2. **`RobustScaler`** *(Recommended & Deployed)*: Centers using the median and scales using the Interquartile Range ($IQR = Q_3 - Q_1$):
   $$z = \frac{x - \text{median}}{IQR}$$
   *Advantage:* Resistant to extreme transaction amounts and outliers, preserving relative distance among normal transactions.
3. **`MinMaxScaler`**: Scales all values to $[0, 1]$:
   $$z = \frac{x - x_{min}}{x_{max} - x_{min}}$$
4. **`MaxAbsScaler`**: Scales by the maximum absolute value into $[-1, 1]$.

### 8.3.2 Non-linear Transformations
- **`PowerTransformer` (Yeo-Johnson)**: Stabilizes variance and maps the heavily right-skewed `Amount` distribution toward a Gaussian normal distribution.
- **`QuantileTransformer`**: Transforms features based on rank quantiles to match a normal distribution.

### Comparison of Transformations on Transaction Amount
![Amount Preprocessing Comparison](figures/amount_scaling_comparison.png)

### 8.3.3 Normalization
- **`Normalizer` (L2 norm)**: Rescales individual sample vectors to unit norm ($\|x\|_2 = 1$).

### 8.3.5 Discretization & Binarization
- **`KBinsDiscretizer`**: Applied to `Time` using uniform strategy to segment continuous seconds into cyclic time buckets across the 48-hour window.
- **`Binarizer`**: Thresholding continuous `Amount` at $\$500.00$ to create a boolean indicator for high-risk transaction volumes.

### 8.3.6 Missing Value Imputation
- Embedded `SimpleImputer(strategy='median')` into the preprocessing pipeline to ensure the pipeline is robust against unexpected missing values in streaming production data.

### 8.3.7 Polynomial Features & 8.3.8 Custom Transformers
- **`PolynomialFeatures(degree=2)`**: Generates interaction terms between key correlated PCA features.
- **`FunctionTransformer(np.log1p)`**: Custom callable transformation for logarithmic scaling.

### Unified `ColumnTransformer` Pipeline
To eliminate **data leakage**, all scalers and transformers are grouped into a Scikit-Learn `ColumnTransformer`. The pipeline is strictly **fitted only on the training set (`X_train`)** and applied to transform downstream test sets:
```python
preprocessor = ColumnTransformer(
    transformers=[
        ("amount", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), ["Amount"]),
        ("time", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), ["Time"]),
        ("pca", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), pca_features)
    ],
    remainder="drop"
)
```

---

## ⚖️ Handling Extreme Class Imbalance
Following the Kaggle tutorial (*Detect Credit Card Frauds with python || Kaggle Projects 2023*), we implement **Random Under-Sampling**:
1. Isolate all $492$ fraudulent transactions (`Class == 1`).
2. Randomly sample $492$ legitimate transactions (`Class == 0`) with a fixed seed (`random_state=42`).
3. Concatenate both subsets into a balanced $50/50$ dataset of $984$ samples.

### Feature Correlation Heatmap on Balanced Dataset
On the balanced dataset, the relationship between PCA features and fraud becomes transparent:
- **Strong Negative Correlation:** $V14$, $V12$, $V10$, $V17$ (lower values strongly indicate fraud).
- **Strong Positive Correlation:** $V2$, $V4$, $V11$, $V19$ (higher values indicate fraud).

![Correlation Matrix](figures/correlation_matrix.png)

---

## 🤖 Model Training & Methodology
We evaluate two complementary machine learning classifiers:
1. **Logistic Regression** *(Primary baseline from video tutorial)*:
   - Linear decision boundary with $L_2$ regularization.
   - Solver: `liblinear`, `max_iter=1000`.
2. **Random Forest Classifier**:
   - Ensemble of $100$ decision trees with `max_depth=10` to capture non-linear interactions.

---

## 📈 Experimental Results & Evaluation

### Performance Comparison on Test Set (Balanced Data)
| Metric | Logistic Regression (Video Baseline) | Random Forest |
| :--- | :---: | :---: |
| **Training Accuracy** | **$94.41\%$** | $99.24\%$ |
| **Test Accuracy** | **$94.92\%$** | $93.40\%$ |
| **Precision** | **$97.83\%$** | $94.74\%$ |
| **Recall (Sensitivity)** | **$91.84\%$** | $91.84\%$ |
| **F1-Score** | **$94.74\%$** | $93.26\%$ |
| **ROC-AUC Score** | **$0.9825$** | $0.9810$ |
| **Average Precision (PR-AUC)** | **$0.9859$** | $0.9836$ |

### Confusion Matrices
![Confusion Matrices](figures/confusion_matrices.png)

### ROC and Precision-Recall Curves
![ROC and PR Curves](figures/roc_pr_curves.png)

---

## 📂 Repository Structure
```
ML-Project/
├── .gitignore                          # Excludes raw 144MB dataset; tracks preprocessed outputs
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation & compliance mappings
├── Data_Preprocessing.ipynb            # Dedicated Data Preprocessing script (Scikit-Learn 8.3)
├── Credit_Card_Fraud_Detection.ipynb   # Complete executed end-to-end ML notebook
├── build_preprocessing_notebook.py     # Script building Data_Preprocessing.ipynb
├── build_notebook.py                   # Script building Credit_Card_Fraud_Detection.ipynb
├── figures/                            # High-resolution figures and evaluation plots
│   ├── amount_scaling_comparison.png
│   ├── class_distribution.png
│   ├── confusion_matrices.png
│   ├── correlation_matrix.png
│   ├── roc_pr_curves.png
│   └── transaction_time_distribution.png
├── data/                               # Cleaned, sample, and preprocessed datasets
│   ├── creditcard_balanced.csv         # 50/50 balanced dataset (984 rows, 504 KB)
│   ├── creditcard_sample.csv           # Stratified sample (5,000 rows, 2.6 MB)
│   ├── X_train_preprocessed.csv        # Preprocessed train features (455 KB)
│   ├── X_test_preprocessed.csv         # Preprocessed test features (114 KB)
│   ├── y_train.csv                     # Train target labels (1.6 KB)
│   └── y_test.csv                      # Test target labels (400 B)
└── src/                                # Modular Python codebase
    ├── preprocessing.py                # Scikit-Learn 8.3 preprocessing pipelines
    ├── model.py                        # Model definitions and evaluation metrics
    └── generate_visualizations.py      # Automated script generating all figures
```

---

## 🚀 How to Run

### Option 1: Run in Google Colab (One-Click)
- **Data Preprocessing Script:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShoumikBalaSomu/ML-Project/blob/main/Data_Preprocessing.ipynb)
- **Full Fraud Detection Pipeline:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShoumikBalaSomu/ML-Project/blob/main/Credit_Card_Fraud_Detection.ipynb)

### Option 2: Run Locally
```bash
git clone https://github.com/ShoumikBalaSomu/ML-Project.git
cd ML-Project
pip install -r requirements.txt
jupyter notebook Data_Preprocessing.ipynb
```

---

## 📚 References
- [Scikit-Learn 8.3: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Detect Credit Card Frauds with python || Kaggle Projects 2023 || Easily Explained (YouTube)](https://www.youtube.com/watch?v=mggdxVcptxU)
- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
