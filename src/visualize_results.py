import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap
import joblib
import os
from evaluate_models import analyze_model_results, analyze_misclassifications
from preprocessing import load_and_preprocess, feature_engineering, select_best_features
from build_models import prepare_data_for_modeling
from load_results import load_model_comparison, get_best_model_name, load_model_results
# Analyze results
def analyze_results(dataset, models, results, best_model, X_test, y_test):
    """Analyze results and visualize performance metrics."""

    df = dataset.copy() 

    # Create a DataFrame for easy comparison
    if isinstance(results, dict):
        print("Results is a dictionary")
        # If results is already a dictionary of model results
        results_df = pd.DataFrame({
            model_name: {
                'accuracy': results[model_name]['accuracy'], 
                'roc_auc': results[model_name]['roc_auc'], 
                'f1': results[model_name]['f1'], 
                'pr_auc': results[model_name]['pr_auc']
            } 
            for model_name in results.keys()
        }).T
    else:
        # If results is a dictionary of DataFrames
        print("Results is a DataFrame")
        results_df = pd.DataFrame(results).T
    
    results_df = results_df.sort_values('pr_auc', ascending=False)
    print("\nModel Performance Comparison:")
    print(results_df)

    # Plot ROC AUC
    print("Plotting ROC AUC")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=results_df.index, y='roc_auc', data=results_df)
    plt.title('ROC AUC Comparison')
    plt.ylabel('ROC AUC')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/roc_auc_comparison.png')
    # plt.show()
    
    # Plot PR AUC
    print("Plotting PR AUC")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=results_df.index, y='pr_auc', data=results_df)
    plt.title('PR AUC Comparison')
    plt.ylabel('PR AUC')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/pr_auc_comparison.png')
    # plt.show()
    
    # Plot F1 Score
    print("Plotting F1 Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=results_df.index, y='f1', data=results_df)
    plt.title('F1 Score Comparison')
    plt.ylabel('F1 Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    # plt.show()

    # Plot Accuracy
    print("Plotting Accuracy")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=results_df.index, y='accuracy', data=results_df)
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    # plt.show()

    # Analyze feature importance
    analyze_model_results(models, X_test, results)
    # Analyze misclassifications
    analyze_misclassifications(best_model, results, X_test, y_test)
    # Visualize misclassifications
    misclassified = X_test.copy()
    misclassified['Actual'] = y_test
    misclassified['Predicted'] = results[best_model]['predictions']
    misclassified['Probability'] = results[best_model]['probabilities']
    misclassified['Misclassified'] = (misclassified['Actual'] != misclassified['Predicted']).astype(int)

    # Plot misclassifications
    print("Plotting misclassifications")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Misclassified', data=misclassified)
    plt.title('Misclassification Count')
    plt.ylabel('Count')
    plt.xlabel('Misclassified')
    # plt.show()

    # Visualize feature importance
    rf_model = models['rf'].named_steps['model']
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Top 20 Important Features (Random Forest)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    # plt.show()
    # Visualize SHAP values
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test)
    except Exception as e:
        print(f"Couldn't create SHAP summary plot due to: {e}")
    # Visualize misclassified samples
    plt.figure(figsize=(10, 6))
    sns.histplot(data=misclassified[misclassified['Misclassified'] == 1], x='Probability', bins=30)
    plt.title('Distribution of Misclassified Samples')
    plt.xlabel('Predicted Probability of Churn')
    plt.ylabel('Count')
    # plt.show()
    # Visualize misclassified samples by feature
    for col in X_test.select_dtypes(include=['int64', 'float64']).columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Misclassified', y=col, data=misclassified)
        plt.title(f'Distribution of {col} by Misclassification')
        plt.xlabel('Misclassified')
        plt.ylabel(col)
        # plt.show()

    # Disctribution of Tenure, Monthly Charges, and Total Charges
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['Tenure Months'], bins=30, kde=True, ax=axes[0])
    axes[0].set_title('Distribution of Tenure (Months)')
    sns.histplot(df['Monthly Charges'], bins=30, kde=True, ax=axes[1])
    axes[1].set_title('Distribution of Monthly Charges')
    sns.histplot(df['Total Charges'], bins=30, kde=True, ax=axes[2])
    axes[2].set_title('Distribution of Total Charges')
    plt.tight_layout()
    # plt.show()

    # Churn Rate by Tenure
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['Churn Label'], y=df['Tenure Months'])
    plt.title('Churn Rate vs. Tenure')
    plt.xlabel('Churn Label (0 = No, 1 = Yes)')
    plt.ylabel('Tenure Months')
    # plt.show()
    # Monthly Charges and Churn
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['Churn Label'], y=df['Monthly Charges'])
    plt.title('Churn Rate vs. Monthly Charges')
    plt.xlabel('Churn Label (0 = No, 1 = Yes)')
    plt.ylabel('Monthly Charges')
    # plt.show()
    # Churn Rate by Contract Type
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Contract', hue='Churn Label', data=df)
    plt.title('Churn Rate by Contract Type')
    plt.xlabel('Contract Type')
    plt.ylabel('Count')
    plt.legend(title='Churn Label', loc='upper right', labels=['No', 'Yes'])
    # plt.show()
    # Churn Rate by Payment Method
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Payment Method', hue='Churn Label', data=df)
    plt.title('Churn Rate by Payment Method')
    plt.xlabel('Payment Method')
    plt.ylabel('Count')
    plt.legend(title='Churn Label', loc='upper right', labels=['No', 'Yes'])
    # plt.show()

# data/customer_churn_datasets/Telco_customer_churn.xlsx

# models
# models/best_churn_model.pkl
# models/lr.pkl
# models/model_results.pkl
# models/rf.pkl
# models/stack.pkl
# models/voting.pkl
# models/xgb.pkl

# results
# results/errors.csv
# results/false_negatives.csv
# results/false_positives.csv
# results/feature_importance.csv
# results/lr_results.csv
# results/misclassified_samples.csv
# results/model_comparison.csv
# results/rf_results.csv
# results/stack_results.csv
# results/voting_results.csv
# results/xgb_results.csv

# Run the analysis
# Load dataset as DataFrame

filepath = "data/customer_churn_datasets/Telco_customer_churn.xlsx"
print("Step 1: Loading and preprocessing data...")
telco_df = load_and_preprocess(filepath)

print("\nStep 2: Applying enhanced feature engineering...")
telco_df_enhanced = feature_engineering(telco_df)
print(f"Dataset shape after feature engineering: {telco_df_enhanced.shape}")

print("\nStep 3: Selecting best features...")
best_features = select_best_features(telco_df_enhanced)

print("\nStep 4: Preparing data for modeling...")
_, X_test, _, y_test = prepare_data_for_modeling(telco_df_enhanced, best_features)

# Load models
models = {model_name: joblib.load(f'models/{model_name}') for model_name in os.listdir('models') if model_name.endswith('.pkl')}

# Load results
results = load_model_results() # dictionary
model_comparison = load_model_comparison()
best_model = get_best_model_name()

# Run the analysis
analyze_results(telco_df_enhanced, models, results, best_model, X_test, y_test)