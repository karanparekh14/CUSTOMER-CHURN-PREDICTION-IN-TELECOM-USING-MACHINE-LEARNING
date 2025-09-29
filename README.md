Customer Churn Prediction in Telecom Using Machine Learning
An Interpretable XGBoost Approach with SHAP Analysis

Author: Karan Tushar Parekh
Program: MS Data Science (MSDS C24 Jan’25)
University: Liverpool John Moores University
Date: October 2025

📖 Project Overview
This repository contains code and data for an interpretable machine learning solution to predict customer churn in the telecommunications industry. The approach centers on:

Optimized XGBoost for high predictive performance

SHAP analysis for global and local interpretability

Business-impact translation including ROI and retention strategy

The final model achieves 87.9% accuracy, with 81.4% precision and 76.3% recall, validated by stakeholders (92% agreement).

📂 Repository Structure
text
├── data/
│   └── Telco_customer_churn.xlsx     # 7,043 customers, 33 features
├── notebooks/
│   └── enhanced_customer_churn.ipynb # Preprocessing, training, interpretability
├── models/
│   └── churn_prediction_model_v1.pkl  # Serialized deployment package
├── README.md                          # Project documentation (this file)
└── requirements.txt                   # Python dependencies
🚀 Getting Started
1. Clone the repository
bash
git clone https://github.com/YourUsername/CUSTOMER-CHURN-PREDICTION-IN-TELECOM-USING-MACHINE-LEARNING.git
cd CUSTOMER-CHURN-PREDICTION-IN-TELECOM-USING-MACHINE-LEARNING
2. Install dependencies
bash
pip install -r requirements.txt
3. Place the dataset
Ensure Telco_customer_churn.xlsx is in the data/ folder.

4. Launch the analysis notebook
Open notebooks/enhanced_customer_churn.ipynb in Jupyter or Google Colab and run all cells.

🔧 Key Steps in the Notebook
Data Loading & Inspection

Load Excel dataset

Display feature overview and churn distribution

Preprocessing & Feature Engineering

Map and rename columns

Convert “Total Charges” and “Churn Label”

Generate 11 new features including CLTV segments, churn scores, and geographic churn rates

Model Preparation & Balancing

Split data into train/test

Standardize numeric features and one-hot encode categoricals (handle unknowns)

Apply SMOTE to address class imbalance

XGBoost Hyperparameter Optimization

RandomizedSearchCV over a large parameter grid

5-fold stratified cross-validation

Baseline Model Comparison

Train Logistic Regression, Decision Tree, Random Forest

Compare metrics to optimized XGBoost

Performance Evaluation

Compute accuracy, precision, recall, F1, AUC

Report confusion matrix and improvement percentages

SHAP Interpretability Analysis

Global feature importance ranking

Business insights mapping top features to churn drivers

SHAP-based customer risk segmentation

Alternative Local Interpretability

Permutation importance fallback when LIME unavailable

Explain individual high-risk predictions

Business Impact & ROI Calculation

Estimate prevented churn, retention value, campaign cost, net benefit, ROI

Stakeholder validation results and recommended retention strategies

Deployment Preparation

Save model and preprocessing pipeline (churn_prediction_model_v1.pkl)

Provide production prediction function with SHAP explanations

Outline next steps: CRM integration, monitoring, A/B testing

📈 Results Summary
Metric	Result	Target
Accuracy	87.9%	87.9%
Precision	81.4%	81.4%
Recall	76.3%	76.3%
AUC	0.94	—
ROI	406%	—
Stakeholder Agreement	92%	—
📦 Dependencies
text
numpy>=1.21
pandas>=1.3
scikit-learn>=1.2
xgboost>=3.0.5
shap>=0.48.0
imbalanced-learn>=0.9
plotly>=5.0
openpyxl
Install via:

bash
pip install -r requirements.txt
📂 Data Description
CustomerID: Unique identifier

Demographics: Gender, SeniorCitizen, Partner, Dependents

Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

Account Info: Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

Business Metrics: CLTV (Customer Lifetime Value), Churn Score, Churn Reason

Geographic: Country, State, City, Zip Code, Latitude, Longitude

🤝 Contributing
Contributions and feedback are welcome! Please open an issue or submit a pull request.

📜 License
This project is licensed under the MIT License.

