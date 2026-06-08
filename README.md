# Titanic Passenger Survival: Statistical Analysis & Predictive Modeling

This repository contains a comprehensive data science project focused on the Titanic passenger dataset. The workflow spans exploratory data analysis (EDA), rigorous parametric hypothesis testing, feature engineering, and evaluating an ensemble machine learning classifier.

## 📂 Project Structure

```text
├── data/
│   ├── titanic.csv                 # Raw dataset
│   └── titanic_processed.csv       # Preprocessed dataset
├── statisticalAnalysis.ipynb       # Exploratory analysis & baseline EDA
├── parametric_hypothesis_testing.py # Script executing parametric statistical tests
├── modeling.ipynb              # Model tuning, grid search, and ensemble evaluation
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## 📊 Core Workflow
### 1. Statistical Analysis & EDA (statisticalAnalysis.ipynb)
Initial deep dive into the dataset to identify core missingness, analyze distributions, and understand class balance. This step includes:

- Aggregating median passenger ages grouped by Passenger Class (Pclass) and Gender (Sex) for strategic missing data imputation.

- Feature grouping and bucket definitions for family variables (SibSpGroup, ParchGroup, FamilySize).


### 2. Parametric Hypothesis Testing (parametric_hypothesis_testing.py)
To validate observations made during EDA, we run parametric statistical tests. This ensures our engineered features contain statistically significant differences regarding target distribution variations before feed-forward processing.

- T-Tests / ANOVA: Implemented to verify if variables like Fare show statistically significant differences across distinct target groups (Survived vs Deceased).

### 3. Model Engineering & Optimization (buildDataset.ipynb)
A multi-model training pipeline optimized via 5-Fold Grid Search Cross-Validation:

- Pre-processing: Categorical features are encoded using one-hot mapping, and numeric data is passed through a scaling structure for relevant architectures.

- Individual Classifiers Evaluated: Decision Tree, Random Forest, Gradient Boosting, and Logistic Regression.

- Ensemble Strategy: Combines individual optimal estimators into a Soft Voting Classifier, generating robust classifications based on averaged prediction probabilities.

## 🚀 Getting Started
**Prerequisites:**

Ensure you have Python 3.8+ installed.


### Installation & Environment Setup
 **1. Clone this repository to your local system:**
 
    git clone [https://github.com/yourusername/titanic-survival-analysis.git](https://github.com/yourusername/titanic-survival-analysis.git)
    cd titanic-survival-analysis

 **2. Set up an isolated virtual environment (recommended):**
 
    python -m venv env
    source env/bin/activate  # On Windows use: env\Scripts\activate

 **3. Install all necessary dependencies:**
 
    pip install -r requirements.txt