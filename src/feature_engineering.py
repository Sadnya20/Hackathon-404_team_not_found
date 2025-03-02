import pandas as pd
import numpy as np
from typing import Tuple, List
import logging
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

class FeatureEngineer:
    def __init__(self):
        self.feature_list = []
        self.pca = None
        self.feature_selector = None
        
    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial-related features"""
        df['Revenue_Cost_Ratio'] = df['Revenue (USD)'] / df['Operating Cost (USD)']
        df['Operating_Margin'] = (
            (df['Revenue (USD)'] - df['Operating Cost (USD)']) / df['Revenue (USD)']
        )
        df['Revenue_per_Seat'] = df['Revenue (USD)'] / df['Load Factor (%)']
        df['Cost_per_Seat'] = df['Operating Cost (USD)'] / df['Load Factor (%)']
        
        return df
        
    def create_operational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create operational-related features"""
        df['Utilization_Efficiency'] = df['Aircraft Utilization (Hours/Day)'] / 24
        df['Asset_Utilization'] = (
            df['Aircraft Utilization (Hours/Day)'] * df['Fleet Availability (%)'] / 100
        )
        df['Maintenance_Ratio'] = (
            df['Maintenance Downtime (Hours)'] / df['Aircraft Utilization (Hours/Day)']
        )
        
        return df
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-related features"""
        df['Peak_Hour'] = df['Scheduled_Departure_Hour'].apply(
            lambda x: 1 if 6 <= x <= 9 or 17 <= x <= 19 else 0
        )
        
        df['Day_Period'] = df['Scheduled_Departure_Hour'].apply(
            lambda x: 'Morning' if 5 <= x < 12
            else 'Afternoon' if 12 <= x < 17
            else 'Evening' if 17 <= x < 22
            else 'Night'
        )
        
        return df
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        df['Load_Utilization'] = df['Load Factor (%)'] * df['Aircraft Utilization (Hours/Day)']
        df['Revenue_per_Hour'] = df['Revenue (USD)'] / df['Aircraft Utilization (Hours/Day)']
        df['Cost_per_Hour'] = df['Operating Cost (USD)'] / df['Aircraft Utilization (Hours/Day)']
        
        return df
        
    def apply_pca(self, df: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction"""
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
            pca_features = self.pca.fit_transform(df)
        else:
            pca_features = self.pca.transform(df)
            
        pca_columns = [f'PCA_{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(pca_features, columns=pca_columns)
        
        return pd.concat([df, df_pca], axis=1)
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
        """Select top k features based on correlation with target"""
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
        else:
            X_selected = self.feature_selector.transform(X)
            
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        return X[selected_features]
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        try:
            # Create basic features
            df = self.create_financial_features(df)
            df = self.create_operational_features(df)
            df = self.create_time_features(df)
            df = self.create_interaction_features(df)
            
            # Store created features
            self.feature_list = df.columns.tolist()
            
            return df
            
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            raise
            
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        return None
