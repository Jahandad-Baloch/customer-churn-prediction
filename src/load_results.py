"""
Utility functions for loading saved model results for post-analysis.
"""
import os
import pandas as pd
import joblib

def load_model_results():
    """
    Load model results from saved files for post-analysis.
    Returns a dictionary with model results and performance metrics.
    """
    
    # loading individual CSV files
    print("Consolidated results file not found, loading from individual CSVs.")
    results = {}
    model_names = ['lr', 'rf', 'xgb', 'stack', 'voting']
    
    for name in model_names:
        metrics_path = f'results/{name}_results.csv'
        preds_path = f'results/{name}_predictions.csv'
        
        if os.path.exists(metrics_path) and os.path.exists(preds_path):
            # Load metrics
            metrics_df = pd.read_csv(metrics_path)
            metrics = dict(zip(metrics_df['Metric'], metrics_df['Value']))
            
            # Load predictions
            preds_df = pd.read_csv(preds_path)
            
            results[name] = {
                'accuracy': metrics.get('accuracy', 0),
                'roc_auc': metrics.get('roc_auc', 0),
                'f1': metrics.get('f1', 0),
                'pr_auc': metrics.get('pr_auc', 0),
                'predictions': preds_df['y_pred'].values,
                'probabilities': preds_df['y_prob'].values
            }
    
    return results

def get_best_model_name(results=None):
    """
    Determine the best model based on PR AUC.
    
    Args:
        results: Optional dict of model results. If None, will load results.
        
    Returns:
        Name of the best performing model
    """
    if results is None:
        results = load_model_results()
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        model_name: {'pr_auc': results[model_name]['pr_auc']}
        for model_name in results.keys()
    }).T
    
    best_model = comparison.sort_values('pr_auc', ascending=False).index[0]
    return best_model

def load_model_comparison():
    """
    Load the model comparison table.
    """
    if os.path.exists('results/model_comparison.csv'):
        return pd.read_csv('results/model_comparison.csv', index_col=0)
    else:
        # Generate comparison from individual results
        results = load_model_results()
        comparison = pd.DataFrame({
            model_name: {
                'accuracy': results[model_name]['accuracy'],
                'roc_auc': results[model_name]['roc_auc'],
                'f1': results[model_name]['f1'],
                'pr_auc': results[model_name]['pr_auc']
            }
            for model_name in results.keys()
        }).T
        return comparison
