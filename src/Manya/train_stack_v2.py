import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

def optimize_f1_threshold(y_true, y_probs):
    best_threshold = 0.5
    best_f1 = 0.0
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
    
    train_path = os.path.join(data_dir, 'train_features_v2.csv')
    test_path = os.path.join(data_dir, 'test_features_v2.csv')
    
    print("Loading V2 engineered data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X = train_df.drop(columns=['exit_status'])
    y = train_df['exit_status']
    
    test_ids = test_df['id']
    X_test = test_df.drop(columns=['id'])
    X_test = X_test[X.columns]
    
    # Calculate scale_pos_weight
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    spw = num_neg / num_pos
    print(f"Calculated scale_pos_weight: {spw:.2f}")
    
    # 5-Fold Setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds_accum = np.zeros(len(X_test))
    
    print(f"Starting {n_splits}-Fold Out-Of-Fold Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, 
                                      scale_pos_weight=spw, random_state=42, eval_metric='logloss')),
            ('lgb', lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, 
                                       scale_pos_weight=spw, random_state=42, verbose=-1))
        ]
        
        if HAS_CATBOOST:
            estimators.append(
                ('cat', CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, 
                                           auto_class_weights='Balanced', verbose=0, random_seed=42))
            )
            
        stack_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
            cv=3,
            n_jobs=-1
        )
        
        print("Training Stacking Ensemble on fold...")
        stack_clf.fit(X_train, y_train)
        
        print("Predicting validation fold...")
        oof_preds[val_idx] = stack_clf.predict_proba(X_val)[:, 1]
        
        print("Predicting test set...")
        test_preds_accum += stack_clf.predict_proba(X_test)[:, 1] / n_splits
        
    print("\nOptimizing F1 Threshold on Full OOF Predictions...")
    optimal_thresh, optimal_f1 = optimize_f1_threshold(y, oof_preds)
    
    print(f"==========================================")
    print(f"Optimal OOF Threshold: {optimal_thresh:.4f}")
    print(f"Full OOF F1 Score: {optimal_f1:.4f}")
    print(f"==========================================")
    
    final_test_preds = (test_preds_accum >= optimal_thresh).astype(int)
    
    submission_df = pd.DataFrame({
        'id': test_ids,
        'exit_status': final_test_preds
    })
    
    submission_path = os.path.join(data_dir, 'submission_v2.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Final V2 submission saved to {submission_path}")
