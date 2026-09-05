import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

def optimize_f1_threshold(y_true, y_probs):
    """Finds the optimal probability threshold to maximize F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0
    
    # Sweep through thresholds
    thresholds = np.linspace(0.1, 0.9, 100)
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = thresh
            
    return best_threshold, best_f1

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    
    train_path = os.path.join(data_dir, 'train_features.csv')
    test_path = os.path.join(data_dir, 'test_features.csv')
    
    print("Loading engineered data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Prepare X and y
    X = train_df.drop(columns=['exit_status'])
    y = train_df['exit_status']
    
    # Keep ID for test submission
    test_ids = test_df['id']
    X_test = test_df.drop(columns=['id'])
    
    # Ensure columns match perfectly
    X_test = X_test[X.columns]
    
    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define Base Learners
    print("Setting up models...")
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss')),
        ('lgb', lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42))
    ]
    
    if HAS_CATBOOST:
        estimators.append(
            ('cat', CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, verbose=0, random_seed=42))
        )
    else:
        print("Warning: CatBoost is not installed. Proceeding without it.")
        
    # Define Stacking Ensemble
    stack_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1
    )
    
    print("Training Stacking Ensemble (this may take a few minutes)...")
    stack_clf.fit(X_train, y_train)
    
    print("Predicting on validation set...")
    val_probs = stack_clf.predict_proba(X_val)[:, 1]
    
    print("Optimizing F1 Threshold...")
    optimal_thresh, optimal_f1 = optimize_f1_threshold(y_val, val_probs)
    
    print(f"==========================================")
    print(f"Optimal Threshold: {optimal_thresh:.4f}")
    print(f"Validation F1 Score: {optimal_f1:.4f}")
    print(f"==========================================")
    
    print("Predicting on test set...")
    test_probs = stack_clf.predict_proba(X_test)[:, 1]
    
    # Apply optimal threshold to test predictions
    test_preds = (test_probs >= optimal_thresh).astype(int)
    
    # Create submission file
    submission_df = pd.DataFrame({
        'id': test_ids,
        'exit_status': test_preds
    })
    
    submission_path = os.path.join(data_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Final submission saved to {submission_path}")
