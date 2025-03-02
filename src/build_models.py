
# Required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# For addressing class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# Step 4: Prepare data for modeling
def prepare_data_for_modeling(telco_df, best_features=None):
    """Prepare and split data for modeling."""
    # Process categorical features
    categorical_columns = telco_df.select_dtypes(include=["object"]).columns
    print(f"Categorical columns for encoding: {list(categorical_columns)}")
    
    # Apply one-hot encoding
    telco_df_encoded = pd.get_dummies(telco_df, columns=categorical_columns, drop_first=True)
    
    # Create feature matrix and target variable
    X = telco_df_encoded.drop(columns=['Churn Label'])
    y = telco_df_encoded['Churn Label']
    
    # Use selected features if provided
    if best_features:
        # Ensure all best_features are in X
        available_features = [col for col in best_features if col in X.columns]
        X = X[available_features]
        print(f"Using {len(available_features)} selected features for modeling")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Class distribution in training set: {np.bincount(y_train)}")
    
    return X_train, X_test, y_train, y_test

# Step 5: Build advanced model with SMOTE and hyperparameter tuning
def build_advanced_model(X_train, y_train):
    """Build an advanced model with SMOTE and hyperparameter tuning."""
    # Identify numerical columns
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numerical_columns)],
        remainder='passthrough'
    )
    
    # Create models with SMOTE addressing class imbalance
    # 1. Logistic Regression with SMOTE
    lr_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', LogisticRegression(random_state=42))
    ])
    
    lr_param_grid = {
        'model__C': [0.01, 0.1, 1, 10],
        'model__solver': ['liblinear', 'saga'],
        'model__max_iter': [2000, 5000]
    }
    
    # 2. Random Forest with SMOTE
    rf_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(random_state=42))
    ])
    
    rf_param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_leaf': [1, 2, 4],
        'model__class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # 3. XGBoost with SMOTE
    xgb_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', xgb.XGBClassifier(random_state=42))
    ])
    
    xgb_param_grid = {
        'model__learning_rate': [0.01, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__n_estimators': [100, 200],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0]
    }
    
    # Perform grid search with cross-validation for each model
    print("Performing grid search for Logistic Regression...")
    lr_grid = GridSearchCV(
        lr_pipeline, lr_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    lr_grid.fit(X_train, y_train)
    
    print("Performing grid search for Random Forest...")
    rf_grid = GridSearchCV(
        rf_pipeline, rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    rf_grid.fit(X_train, y_train)
    
    print("Performing grid search for XGBoost...")
    xgb_grid = GridSearchCV(
        xgb_pipeline, xgb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    xgb_grid.fit(X_train, y_train)
    
    # Get best models
    best_lr = lr_grid.best_estimator_
    best_rf = rf_grid.best_estimator_
    best_xgb = xgb_grid.best_estimator_
    
    print(f"Best Logistic Regression params: {lr_grid.best_params_}")
    print(f"Best Random Forest params: {rf_grid.best_params_}")
    print(f"Best XGBoost params: {xgb_grid.best_params_}")
    
    # Create a stacking ensemble
    base_models = [
        ('lr', best_lr),
        ('rf', best_rf),
        ('xgb', best_xgb)
    ]
    
    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    # Create a voting ensemble
    voting = VotingClassifier(
        estimators=base_models,
        voting='soft'
    )
    
    # Train the ensemble models
    print("Training stacking ensemble...")
    stack.fit(X_train, y_train)
    
    print("Training voting ensemble...")
    voting.fit(X_train, y_train)
    
    # Return all models
    models = {
        'lr': best_lr,
        'rf': best_rf,
        'xgb': best_xgb,
        'stack': stack,
        'voting': voting
    }
    
    return models

