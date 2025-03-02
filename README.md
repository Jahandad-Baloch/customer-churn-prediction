# 📊 Customer Churn Prediction

An End-to-End Machine Learning Pipeline for Predictive Analytics

## **Overview**
This project is an **end-to-end Machine Learning pipeline** for **predicting customer churn** using a **Telco Customer Churn dataset**. It covers **data preprocessing, feature engineering, model training, evaluation, and results visualization**.

## **🔧 Features**
- **Advanced feature engineering** (temporal, financial, segmentation-based features).
- **SMOTE-based class imbalance handling** for better churn prediction.
- **Multiple machine learning models**, including:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Stacking & Voting Ensembles
- **Performance evaluation with AUC, F1-score, PR AUC**.
- **Misclassification analysis** to understand false positives/negatives.
- **Feature importance analysis using SHAP and Random Forest**.

## **📁 Project Structure**
```
📂 customer-churn-prediction
│── 📂 data
│   └── 📂 customer_churn_datasets
│       └── 📄 Telco_customer_churn.xlsx  # Dataset file
│── 📂 models                            # Saved models
│   ├── 📄 best_churn_model.pkl
│   ├── 📄 lr.pkl
│   ├── 📄 model_results.pkl
│   ├── 📄 rf.pkl
│   ├── 📄 stack.pkl
│   ├── 📄 voting.pkl
│   ├── 📄 xgb.pkl
│── 📂 results                           # Model evaluation outputs
│   ├── 📄 best_model.txt
│   ├── 📄 errors.csv
│   ├── 📄 false_negatives.csv
│   ├── 📄 false_positives.csv
│   ├── 📄 feature_importance.csv
│   ├── 📄 lr_predictions.csv
│   ├── 📄 lr_results.csv
│   ├── 📄 misclassified_samples.csv
│   ├── 📄 model_comparison.csv
│   ├── 📄 model_results.pkl
│   ├── 📄 rf_predictions.csv
│   ├── 📄 rf_results.csv
│   ├── 📄 stack_predictions.csv
│   ├── 📄 stack_results.csv
│   ├── 📄 test_data.csv
│   ├── 📄 voting_predictions.csv
│   ├── 📄 voting_results.csv
│   ├── 📄 xgb_predictions.csv
│   ├── 📄 xgb_results.csv
│── 📂 src                               # Project source code
│   ├── 📄 __init__.py                   # Package initialization
│   ├── 📄 build_models.py               # Model training script
│   ├── 📄 evaluate_models.py            # Model evaluation & metrics calculation
│   ├── 📄 load_results.py               # Utility to load saved model results
│   ├── 📄 preprocessing.py              # Data preprocessing functions
│   ├── 📄 train_models.py               # Main script to train models
│   ├── 📄 visualize_results.py          # Visualization script for results
│── 📄 README.md                         # Project documentation
│── 📄 requirements.txt                  # Python dependencies
```

## **🚀 Installation & Setup**

### Clone the repository
```bash
git clone https://github.com/Jahandad-Baloch/customer-churn-prediction.git
cd customer-churn-prediction
```

### Create and activate a virtual environment (optional but recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Mac/Linux
.venv\Scripts\activate     # On Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the full pipeline
```bash
python src/train_models.py
```

## **📊 Results**

### Best Model: Stacking Ensemble
**Performance Summary:**
- Accuracy: 79.4%
- ROC AUC: 84.9%
- PR AUC: 65.7%

The best model is saved as `models/best_churn_model.pkl`

## **📈 Visualizations**
- Feature importance (SHAP & RF)
- Misclassification analysis
- Churn rate distribution
- Model comparison (PR AUC, ROC AUC, F1-score)

## **📌 Future Improvements**
- Integration of deep learning models
- API & cloud deployment (FastAPI + Docker)
- Real-time churn prediction system

## **🔗 Connect**
- **Author**: Jahandad Baloch
- **Github**: [@Jahandad-Baloch](https://github.com/Jahandad-Baloch)
- **Email**: jahandadbaloch@gmail.com

