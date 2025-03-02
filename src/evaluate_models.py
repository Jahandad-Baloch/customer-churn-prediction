import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, precision_recall_curve, auc
import shap # For model interpretability
import matplotlib.pyplot as plt
import os
import joblib

# Step 6: Model evaluation with advanced metrics
def evaluate_models(models, X_test, y_test):
    """Evaluate models using multiple metrics, especially for imbalanced data."""
    results = {}
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        # Precision-Recall AUC is better for imbalanced data
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1': f1,
            'pr_auc': pr_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Create proper DataFrame for results
        metrics_df = pd.DataFrame({
            'Metric': ['accuracy', 'roc_auc', 'f1', 'pr_auc'],
            'Value': [accuracy, roc_auc, f1, pr_auc]
        })
        
        # Save metrics to CSV
        metrics_df.to_csv(f'results/{name}_results.csv', index=False)
        
        # Save predictions separately for later analysis
        pred_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_pred_proba
        })
        pred_df.to_csv(f'results/{name}_predictions.csv', index=False)

        # Print classification report
        print(f"\n{name.upper()} PERFORMANCE:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print(classification_report(y_test, y_pred)) 

    # Create a consolidated comparison DataFrame
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
    
    return results


# Step 7: Feature importance and model interpretation
def analyze_model_results(models, X_test, results):
    """Analyze model results and feature importance."""
    # Get feature importance for Random Forest
    rf_model = models['rf'].named_steps['model']
    
    # Create SHAP explainer for the Random Forest model
    try:
        # Extract feature names (might require some transformation)
        feature_names = X_test.columns.tolist()
        
        # Assuming SHAP works with our pipeline
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)
        
        # Average absolute SHAP values per feature
        shap_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(shap_values).mean(axis=0)
        })
        
        # Sort by importance
        shap_importance = shap_importance.sort_values('Importance', ascending=False).head(20)

        # Save SHAP values
        shap_importance.to_csv('results/shap_importance.csv', index=False)
        # Plot SHAP summary
        shap.summary_plot(shap_values, X_test)
        # Save SHAP summary plot
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig('results/shap_summary_plot.png')
        plt.close()
        # Save SHAP values plot
        shap.dependence_plot(0, shap_values, X_test)
        plt.savefig('results/shap_dependence_plot.png')
        plt.close()

        # Print top 20 important features
        print("\nTop 20 Important Features (SHAP values):")
        print(shap_importance)
        
    except Exception as e:
        print(f"Couldn't create SHAP explanation due to: {e}")
        
        # Fallback to feature importance from Random Forest
        if hasattr(rf_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': rf_model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False).head(20)

            # Save feature importance
            feature_importance.to_csv('results/feature_importance.csv', index=False)

            # Print top 20 important features
            print("\nTop 20 Important Features (Random Forest):")
            print(feature_importance)
    
    # Compare model performances
    model_comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'ROC AUC': [results[model]['roc_auc'] for model in results],
        'F1 Score': [results[model]['f1'] for model in results],
        'PR AUC': [results[model]['pr_auc'] for model in results]
    })

    # Save model comparison results
    model_comparison.to_csv('results/model_comparison.csv', index=False)

    # Print model comparison    
    print("\nModel Performance Comparison:")
    print(model_comparison.sort_values('PR AUC', ascending=False))
    
    return model_comparison

# Step 8: Identify misclassification patterns
def analyze_misclassifications(best_model_name, results, X_test, y_test):
    """Analyze patterns in misclassified samples."""
    # Get predictions from the best model
    y_pred = results[best_model_name]['predictions']
    
    # Identify misclassified samples
    misclassified = X_test.copy()
    misclassified['Actual'] = y_test
    misclassified['Predicted'] = y_pred
    misclassified['Probability'] = results[best_model_name]['probabilities']
    misclassified['Misclassified'] = (misclassified['Actual'] != misclassified['Predicted']).astype(int)
    
    # Focus on misclassified samples
    errors = misclassified[misclassified['Misclassified'] == 1]
    
    # False Negatives (customers who churned but we predicted they wouldn't)
    false_negatives = errors[errors['Actual'] == 1]
    
    # False Positives (customers who didn't churn but we predicted they would)
    false_positives = errors[errors['Actual'] == 0]

    # Save misclassification statistics
    misclassified.to_csv('results/misclassified_samples.csv', index=False)
    # Save false negatives and false positives
    false_negatives.to_csv('results/false_negatives.csv', index=False)
    false_positives.to_csv('results/false_positives.csv', index=False)
    # Save overall errors
    errors.to_csv('results/errors.csv', index=False)
    # Save false negatives
    false_negatives.to_csv('results/false_negatives.csv', index=False)
    # Save false positives
    false_positives.to_csv('results/false_positives.csv', index=False)

    # Print misclassification statistics    
    print(f"\nTotal test samples: {len(X_test)}")
    print(f"Misclassified samples: {len(errors)} ({len(errors)/len(X_test):.2%} of test data)")
    print(f"False Negatives: {len(false_negatives)} ({len(false_negatives)/sum(y_test):.2%} of actual churners)")
    print(f"False Positives: {len(false_positives)} ({len(false_positives)/(len(y_test)-sum(y_test)):.2%} of actual non-churners)")
    
    # Optional: Analyze patterns in misclassifications
    if len(errors) > 0:
        print("\nPatterns in misclassified samples (compared to correctly classified):")
        # For numerical features
        for col in X_test.select_dtypes(include=['int64', 'float64']).columns:
            correct_mean = misclassified[misclassified['Misclassified'] == 0][col].mean()
            error_mean = misclassified[misclassified['Misclassified'] == 1][col].mean()
            diff_pct = abs(error_mean - correct_mean) / (correct_mean + 0.0001) * 100
            
            if diff_pct > 10:  # Only show features with >10% difference
                print(f"{col}: Correct mean = {correct_mean:.2f}, Error mean = {error_mean:.2f}, Diff = {diff_pct:.1f}%")
        
        # For categorical features
        categorical_cols = X_test.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            correct_counts = misclassified[misclassified['Misclassified'] == 0][col].value_counts(normalize=True)
            error_counts = misclassified[misclassified['Misclassified'] == 1][col].value_counts(normalize=True)
            diff_pct = abs(error_counts - correct_counts).max() * 100
            
            if diff_pct > 10:
                print(f"{col}: Correct distribution = {correct_counts}, Error distribution = {error_counts}, Max Diff = {diff_pct:.1f}%")

    return errors