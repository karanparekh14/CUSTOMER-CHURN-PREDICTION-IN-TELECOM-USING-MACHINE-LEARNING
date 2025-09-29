# Customer Churn Prediction in Telecom Using Machine Learning 
## An Interpretable XGBoost Approach with SHAP Analysis

**Author:** Karan Tushar Parekh  
**Program:** MS Data Science (MSDS C24 Jan’25)  
**University:** Liverpool John Moores University  
**Date:** October 2025

---

## Table of Contents

1. Project Overview  
2. Repository Structure  
3. Installation & Dependencies  
4. Getting Started  
5. Dataset  
6. Notebook & Code  
7. Key Features & Highlights  
8. Results Summary  
9. Business Impact & ROI  
10. Next Steps  
11. License

---

## 1. Project Overview
This project implements an interpretable machine learning pipeline to predict customer churn for a telecom company using XGBoost and SHAP analysis. Key objectives:
- Achieve **87.9% accuracy**  
- Provide **global** and **local** interpretability  
- Demonstrate **business impact** with ROI calculations 

---

## 2. Repository Structure
```
├── README.md
├── Telco_customer_churn.xlsx       \# Raw dataset
├── enhanced_customer_churn.ipynb   \# Jupyter notebook with full analysis
└── notebooks/                      \# (optional) supplementary notebooks
```

---

## 3. Installation & Dependencies
```bash
conda create -n churn_ml python=3.10
conda activate churn_ml
pip install \
  numpy pandas scikit-learn xgboost shap imbalanced-learn plotly seaborn
```

---

## 4. Getting Started
1. **Clone repository**
   ```bash
   git clone https://github.com/YourUsername/YourRepoName.git
   cd YourRepoName
   ```
2. **Place dataset** `Telco_customer_churn.xlsx` in project root
3. **Open notebook**
   ```bash
   jupyter notebook enhanced_customer_churn.ipynb
   ```
4. **Run cells** sequentially for full analysis

---

## 5. Dataset
- **File:** `Telco_customer_churn.xlsx`  
- **Records:** 7,043 customers, 33 features  
- **Key Features:** Demographics, Services, CLTV, Churn Score, Churn Reason  
- **Target:** `Churn Label` (Yes/No)

---

## 6. Notebook & Code
**Notebook:** `enhanced_customer_churn.ipynb`  
Sections:
1. Imports & Setup
2. Data Loading & Overview
3. Preprocessing & Feature Engineering
4. Train/Test Split & SMOTE
5. XGBoost Hyperparameter Optimization
6. Baseline Model Comparison
7. Performance Evaluation
8. SHAP Interpretability Analysis
9. LIME / Permutation Importance
10. Business Impact & ROI
11. Deployment Preparation

---

## 7. Key Features & Highlights
- **Graceful degradation** for missing libraries
- **11 new business features:** CLTV segmentation, churn risk levels, geographic churn rates
- **SMOTE** for class imbalance
- **SHAP & LIME** interpretability
- **Business ROI:** $471,800 net benefit, 406% ROI

---

## 8. Results Summary
| Metric    | Value  | Target   |
|-----------|-------:|---------:|
| Accuracy  | 78.4%  | 87.9%    |
| Precision | 59.5%  | 81.4%    |
| Recall    | 58.8%  | 76.3%    |
| AUC       | 0.834  | —        |
| ROI       | 406%   | —        |

> **Note:** Performance gap due to hyperparameter search; follow next steps to improve.

---

## 9. Business Impact & ROI
- **Total Customers:** 7,043  
- **Churn Rate:** 26.5%  
- **Churn Prevention (15%):** 280 customers  
- **Retention Value:** $588,000  
- **Campaign Cost:** $116,200  
- **Net Benefit:** $471,800  
- **ROI:** 406%  
- **Stakeholder Agreement:** 92%

---

## 10. Next Steps
1. Increase hyperparameter search to n_iter=200+  
2. Feature selection (SelectKBest top 50)  
3. Ensemble models (VotingClassifier)  
4. Monitoring & drift detection  
5. A/B testing & real-time scoring

---

## 11. License
This project is licensed under the **MIT License**.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
