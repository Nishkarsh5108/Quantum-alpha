import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class QuantumAlphaImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.credit_score_median_ = None
        self.acc_balance_medians_ = None
        self.global_acc_balance_median_ = None
        self.prod_count_model_ = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Encoders for the prod_count prediction model
        self.country_encoder_ = LabelEncoder()
        self.gender_encoder_ = LabelEncoder()

    def fit(self, X, y=None):
        X_fit = X.copy()
        
        # 1. Country
        X_fit['country'] = X_fit['country'].fillna('Unknown')
        
        # 2. Credit Score
        self.credit_score_median_ = X_fit['credit_score'].median()
        X_fit['credit_score'] = X_fit['credit_score'].fillna(self.credit_score_median_)
        
        # 3. Account Balance
        self.acc_balance_medians_ = X_fit.groupby('country')['acc_balance'].median()
        self.global_acc_balance_median_ = X_fit['acc_balance'].median()
        
        X_fit['acc_balance'] = X_fit.apply(
            lambda row: self.acc_balance_medians_.get(row['country'], self.global_acc_balance_median_) 
            if pd.isna(row['acc_balance']) else row['acc_balance'],
            axis=1
        )
        
        # 4. Product Count (Train Model)
        # Prepare data for prod_count model
        X_prod = X_fit.copy()
        
        # Fit encoders on available categories
        # Note: In practice, Gender has no NaNs based on the snippet, but just in case we fill it
        X_prod['gender'] = X_prod['gender'].fillna('Unknown')
        
        self.country_encoder_.fit(X_prod['country'])
        self.gender_encoder_.fit(X_prod['gender'])
        
        X_prod['country_encoded'] = self.country_encoder_.transform(X_prod['country'])
        X_prod['gender_encoded'] = self.gender_encoder_.transform(X_prod['gender'])
        
        # Features to use for predicting prod_count
        features = [
            'credit_score', 'country_encoded', 'gender_encoded', 'age', 
            'tenure', 'acc_balance', 'has_card', 'is_active', 'estimated_salary'
        ]
        
        # Filter rows where prod_count is not null to train the model
        train_data = X_prod.dropna(subset=['prod_count'])
        
        X_train_model = train_data[features]
        y_train_model = train_data['prod_count']
        
        self.prod_count_model_.fit(X_train_model, y_train_model)
        
        return self

    def transform(self, X):
        X_trans = X.copy()
        
        # 1. Country
        X_trans['country'] = X_trans['country'].fillna('Unknown')
        
        # 2. Credit Score
        X_trans['credit_score'] = X_trans['credit_score'].fillna(self.credit_score_median_)
        
        # 3. Account Balance
        X_trans['acc_balance'] = X_trans.apply(
            lambda row: self.acc_balance_medians_.get(row['country'], self.global_acc_balance_median_) 
            if pd.isna(row['acc_balance']) else row['acc_balance'],
            axis=1
        )
        
        # 4. Product Count (Predict missing)
        missing_prod_mask = X_trans['prod_count'].isna()
        
        if missing_prod_mask.any():
            X_pred = X_trans[missing_prod_mask].copy()
            
            # Handle any unseen categories safely
            X_pred['gender'] = X_pred['gender'].fillna('Unknown')
            
            # Map known categories or default to a safe value (like 0) if unseen
            known_countries = set(self.country_encoder_.classes_)
            X_pred['country'] = X_pred['country'].apply(lambda x: x if x in known_countries else 'Unknown')
            
            # Re-fit encoders to include 'Unknown' if it wasn't there (edge case handled poorly by LabelEncoder)
            # A safer way is to map classes directly, but since we fit 'Unknown' in fit(), it should be fine.
            # Using a custom mapping to handle unseen classes in test data:
            country_mapping = {c: i for i, c in enumerate(self.country_encoder_.classes_)}
            gender_mapping = {c: i for i, c in enumerate(self.gender_encoder_.classes_)}
            
            # If a new category appears in test, we map it to -1 or a default, but let's assume 'Unknown' exists.
            X_pred['country_encoded'] = X_pred['country'].map(lambda x: country_mapping.get(x, 0)) 
            X_pred['gender_encoded'] = X_pred['gender'].map(lambda x: gender_mapping.get(x, 0))
            
            features = [
                'credit_score', 'country_encoded', 'gender_encoded', 'age', 
                'tenure', 'acc_balance', 'has_card', 'is_active', 'estimated_salary'
            ]
            
            predictions = self.prod_count_model_.predict(X_pred[features])
            X_trans.loc[missing_prod_mask, 'prod_count'] = predictions
            
        return X_trans

if __name__ == "__main__":
    import os
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate target variable if you don't want it passed around, 
    # though the Imputer ignores it explicitly.
    # We will just pass the whole dataframe to the imputer.
    
    imputer = QuantumAlphaImputer()
    
    print("Fitting imputer on train data...")
    imputer.fit(train_df)
    
    print("Transforming train data...")
    train_clean = imputer.transform(train_df)
    
    print("Transforming test data...")
    test_clean = imputer.transform(test_df)
    
    # Verify no NaNs remain in target columns
    print("\nRemaining NaNs in Train:")
    print(train_clean[['credit_score', 'country', 'acc_balance', 'prod_count']].isna().sum())
    
    # Save the cleaned datasets
    train_clean_path = os.path.join(data_dir, 'train_clean.csv')
    test_clean_path = os.path.join(data_dir, 'test_clean.csv')
    
    print(f"\nSaving cleaned train data to {train_clean_path}...")
    train_clean.to_csv(train_clean_path, index=False)
    
    print(f"Saving cleaned test data to {test_clean_path}...")
    test_clean.to_csv(test_clean_path, index=False)
    
    print("Done!")
