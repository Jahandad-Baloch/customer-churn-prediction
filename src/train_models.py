import joblib
import os
import pandas as pd

from preprocessing import load_and_preprocess, feature_engineering, select_best_features
from build_models import prepare_data_for_modeling, build_advanced_model
from evaluate_models import evaluate_models, analyze_model_results, analyze_misclassifications


# Main function to run the entire pipeline
def run_churn_prediction_pipeline(filepath):
    """Run the complete churn prediction pipeline with all enhancements."""
    print("Starting enhanced churn prediction pipeline...\n")
    
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    telco_df = load_and_preprocess(filepath)
    
    # Step 2: Apply feature engineering
    print("\nStep 2: Applying enhanced feature engineering...")
    telco_df_enhanced = feature_engineering(telco_df)
    print(f"Dataset shape after feature engineering: {telco_df_enhanced.shape}")
    
    # Step 3: Select best features
    print("\nStep 3: Selecting best features...")
    best_features = select_best_features(telco_df_enhanced, 'Churn Label', top_n=30)
    
    # Step 4: Prepare data for modeling
    print("\nStep 4: Preparing data for modeling...")
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(telco_df_enhanced, best_features)
    
    # Step 5: Build advanced models
    print("\nStep 5: Building advanced models with SMOTE and hyperparameter tuning...")
    models = build_advanced_model(X_train, y_train)
    
    # Step 6: Evaluate models
    print("\nStep 6: Evaluating models...")
    results = evaluate_models(models, X_test, y_test)
    
    # Step 7: Analyze model results
    print("\nStep 7: Analyzing model results and feature importance...")
    model_comparison = analyze_model_results(models, X_test, results)
    
    # Step 8: Analyze misclassifications
    print("\nStep 8: Analyzing misclassification patterns...")
    # Determine best model based on PR AUC (suitable for imbalanced data)
    best_model_name = model_comparison.sort_values('PR AUC', ascending=False).iloc[0]['Model']
    print(f"Best model based on PR AUC: {best_model_name}")
    errors = analyze_misclassifications(best_model_name, results, X_test, y_test)
    
    # Save the test data for future analysis
    test_data = pd.DataFrame(X_test)
    test_data['y_true'] = y_test
    test_data.to_csv('results/test_data.csv', index=False)
    
    print("\nChurn prediction pipeline completed successfully!")
    return models, results, best_model_name

# Run the pipeline
if __name__ == "__main__":
    models, results, best_model = run_churn_prediction_pipeline("data/customer_churn_datasets/Telco_customer_churn.xlsx")

    # Save models
    for model_name, model in models.items():
        joblib.dump(model, f'models/{model_name}.pkl')
        print(f"Saved model: models/{model_name}.pkl")

    # Create and save a consolidated comparison DataFrame
    comparison_df = pd.DataFrame({
        model_name: {
            'accuracy': results[model_name]['accuracy'],
            'roc_auc': results[model_name]['roc_auc'],
            'f1': results[model_name]['f1'],
            'pr_auc': results[model_name]['pr_auc']
        }
        for model_name in results.keys()
    }).T
    
    comparison_df.to_csv('results/model_comparison.csv')
    print("Saved model comparison to: results/model_comparison.csv")

    # Save the best model
    joblib.dump(models[best_model], 'models/best_churn_model.pkl')
    
    # Save the best model name for reference
    with open('results/best_model.txt', 'w') as f:
        f.write(best_model)
    
    print(f"Saved best model ({best_model}) to: models/best_churn_model.pkl")