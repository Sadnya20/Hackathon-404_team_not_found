import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load and preprocess the airline data"""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Preprocess the data"""
    # Add your preprocessing steps here
    return df

def split_data(df, target_column='Profit'):
    """Split features and target"""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y
