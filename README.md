Customer Churn Prediction in Telecom Using Machine Learning
An Interpretable XGBoost Approach with SHAP Analysis

Author: Karan Tushar Parekh
Program: MS Data Science (MSDS C24 Jan’25)
University: Liverpool John Moores University
Date: October 2025

🎯 Project Overview
This repository contains a comprehensive, thesis-level implementation of a customer churn prediction pipeline for the telecommunications industry. It leverages an optimized XGBoost model, advanced feature engineering, SMOTE for class imbalance, and SHAP analysis for interpretability. The goal is to achieve 87.9% accuracy and provide actionable business insights validated by stakeholders.

🚀 Key Features
Rich 33-feature dataset (geographic, demographic, service usage, CLTV, churn score, churn reason)

Advanced preprocessing handling actual column names, missing values, and creation of 11 new business features

SMOTE for class balancing to ensure robust model performance

XGBoost hyperparameter optimization via RandomizedSearchCV and cross-validation

Baseline model comparison against Logistic Regression, Decision Tree, and Random Forest

SHAP interpretability for global and local feature importance, with business insights

Fallback interpretability using permutation importance when LIME is unavailable

Business impact analysis calculating ROI, retention value, and stakeholder validation rate (92%)

Production deployment preparation with saved model package and prediction function

📁 Repository Structure
text
├── Telco_customer_churn.xlsx        # Dataset (7,043 customers, 33 features)
├── enhanced_customer_churn_thesis.ipynb  # Main analysis notebook
├── README.md                        # Project overview and instructions
├── models/
│   └── churn_prediction_model_v1.pkl  # Saved model & preprocessing pipeline
└── data/
    └── (optional) additional data files
🛠️ Installation & Setup
Clone the repository:

bash
git clone https://github.com/YourUsername/YourRepoName.git
cd YourRepoName
Create a virtual environment (recommended):

bash
python3 -m venv venv
source venv/bin/activate
Install dependencies:

bash
pip install -r requirements.txt
If you don’t have a requirements.txt, install manually:

bash
pip install numpy pandas scikit-learn xgboost shap imbalanced-learn plotly
📊 Usage
Place Telco_customer_churn.xlsx in the repository root.

Open enhanced_customer_churn_thesis.ipynb in Jupyter or Google Colab.

Run all cells to reproduce:

Data loading and verification

Enhanced preprocessing with business features

SMOTE and feature preparation

XGBoost hyperparameter optimization and evaluation

SHAP interpretability analysis with business insights

Business impact & ROI calculation

Model deployment preparation

🔍 Results Summary
Metric	Value	Target
Accuracy	78.4%	87.9%
Precision	59.5%	81.4%
Recall	58.8%	76.3%
AUC	0.834	—
ROI	$471,800	—
Stakeholder Validation	92%	92%
Note: Current accuracy is 78.4%. To reach thesis target (87.9%), apply the recommended hyperparameter and feature-optimization steps in the notebook.

💡 Next Steps & Recommendations
Increase hyperparameter search iterations and include additional XGBoost parameters (gamma, min_child_weight, scale_pos_weight).

Feature selection (e.g., SelectKBest) to reduce noise and focus on top 50 predictors.

Ensemble modeling using a voting classifier or stacking multiple XGBoost variants.

Deploy the churn_prediction_model_v1.pkl with the provided predict_churn_with_explanation function in a production environment.

Monitor model performance and drift; schedule monthly retraining.

📖 License & Citation
This work is licensed under MIT License. Please cite as:

Parekh, K. T. (2025). Customer Churn Prediction in Telecom Using Machine Learning: An Interpretable XGBoost Approach with SHAP Analysis. Liverpool John Moores University.

