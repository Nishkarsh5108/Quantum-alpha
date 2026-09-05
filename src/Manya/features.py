import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold

def create_advanced_features(df, is_train=True, target_col='exit_status'):
    """Creates advanced ratio and interaction features."""
    df_feat = df.copy()
    
    # Financial Ratios
    # Add a small epsilon to denominator to prevent division by zero
    epsilon = 1e-5
    df_feat['balance_to_salary_ratio'] = df_feat['acc_balance'] / (df_feat['estimated_salary'] + epsilon)
    df_feat['credit_to_age_ratio'] = df_feat['credit_score'] / (df_feat['age'] + epsilon)
    df_feat['age_to_tenure_ratio'] = df_feat['age'] / (df_feat['tenure'] + epsilon)
    df_feat['products_per_tenure'] = df_feat['prod_count'] / (df_feat['tenure'] + epsilon)
    
    # Active status interaction
    df_feat['active_by_credit'] = df_feat['is_active'] * df_feat['credit_score']
    
    return df_feat

def target_encode_last_name(train_df, test_df, col='last_name', target='exit_status', n_splits=5):
    """
    Performs K-Fold Target Encoding on a high-cardinality categorical feature (last_name).
    This extracts the signal without overfitting (preventing data leakage).
    """
    # Create empty columns
    train_df[f'{col}_te'] = np.nan
    test_df[f'{col}_te'] = np.nan
    
    # Calculate global mean for unseen categories
    global_mean = train_df[target].mean()
    
    # K-Fold Target Encoding for Train
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train_df):
        X_tr, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
        
        # Calculate mean target per category in the current fold's training set
        means = X_tr.groupby(col)[target].mean()
        
        # Map the means to the validation set
        train_df.loc[val_idx, f'{col}_te'] = train_df.loc[val_idx, col].map(means)
        
    # Fill any remaining NaNs in train (categories that only appeared in validation fold) with global mean
    train_df[f'{col}_te'] = train_df[f'{col}_te'].fillna(global_mean)
    
    # Target Encoding for Test (using the whole train set)
    full_means = train_df.groupby(col)[target].mean()
    test_df[f'{col}_te'] = test_df[col].map(full_means)
    
    # Fill unseen categories in test with global mean
    test_df[f'{col}_te'] = test_df[f'{col}_te'].fillna(global_mean)
    
    return train_df, test_df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    
    train_path = os.path.join(data_dir, 'train_clean.csv')
    test_path = os.path.join(data_dir, 'test_clean.csv')
    
    print("Loading cleaned data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print("Engineering ratio features...")
    train_feat = create_advanced_features(train_df, is_train=True)
    test_feat = create_advanced_features(test_df, is_train=False)
    
    print("Applying K-Fold Target Encoding to 'last_name'...")
    train_feat, test_feat = target_encode_last_name(train_feat, test_feat)
    
    # Drop columns that are no longer needed for modeling
    cols_to_drop = ['id', 'customer_id', 'last_name']
    
    train_final = train_feat.drop(columns=cols_to_drop)
    
    # Keep 'id' in test set because we'll need it for the submission file
    test_final = test_feat.drop(columns=['customer_id', 'last_name'])
    
    # Encode Gender
    train_final['gender'] = train_final['gender'].map({'Male': 1, 'Female': 0, 'Unknown': 2}).fillna(2)
    test_final['gender'] = test_final['gender'].map({'Male': 1, 'Female': 0, 'Unknown': 2}).fillna(2)
    
    # Encode Country (One-Hot)
    print("One-hot encoding country...")
    train_final = pd.get_dummies(train_final, columns=['country'], prefix='country')
    test_final = pd.get_dummies(test_final, columns=['country'], prefix='country')
    
    # Ensure test has all columns train has (except exit_status)
    for col in train_final.columns:
        if col not in test_final.columns and col != 'exit_status':
            test_final[col] = 0
            
    train_final_path = os.path.join(data_dir, 'train_features.csv')
    test_final_path = os.path.join(data_dir, 'test_features.csv')
    
    print(f"Saving final engineered datasets to {train_final_path} and {test_final_path}...")
    train_final.to_csv(train_final_path, index=False)
    test_final.to_csv(test_final_path, index=False)
    
    print("Feature Engineering Complete!")
    print(f"Train shape: {train_final.shape}")
    print(f"Test shape: {test_final.shape}")
