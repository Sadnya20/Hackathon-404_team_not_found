import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import logging

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        self.predictions = None
        self.actual_values = None
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate various performance metrics"""
        try:
            metrics = {
                'r2_score': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
            }
            
            self.metrics = metrics
            self.predictions = y_pred
            self.actual_values = y_true
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            raise
            
    def plot_actual_vs_predicted(self, save_path: str = None) -> None:
        """Plot actual vs predicted values"""
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.actual_values, self.predictions, alpha=0.5)
            plt.plot([self.actual_values.min(), self.actual_values.max()],
                    [self.actual_values.min(), self.actual_values.max()],
                    'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting actual vs predicted: {e}")
            raise
            
    def plot_residuals(self, save_path: str = None) -> None:
        """Plot residuals analysis"""
        try:
            residuals = self.actual_values - self.predictions
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Residuals vs Predicted
            ax1.scatter(self.predictions, residuals, alpha=0.5)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('Predicted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residuals vs Predicted Values')
            
            # Residuals Distribution
            sns.histplot(residuals, ax=ax2)
            ax2.set_title('Residuals Distribution')
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting residuals: {e}")
            raise
            
    def plot_feature_importance(self, model: Any, feature_names: list,
                              save_path: str = None) -> None:
        """Plot feature importance"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(12, 6))
                plt.title('Feature Importances')
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)),
                          [feature_names[i] for i in indices],
                          rotation=45)
                
                if save_path:
                    plt.savefig(save_path)
                plt.close()
                
        except Exception as e:
            logging.error(f"Error plotting feature importance: {e}")
            raise
            
    def generate_evaluation_report(self, save_path: str = None) -> Dict:
        """Generate comprehensive evaluation report"""
        report = {
            'metrics': self.metrics,
            'prediction_summary': {
                'mean_prediction': np.mean(self.predictions),
                'std_prediction': np.std(self.predictions),
                'min_prediction': np.min(self.predictions),
                'max_prediction': np.max(self.predictions)
            },
            'actual_summary': {
                'mean_actual': np.mean(self.actual_values),
                'std_actual': np.std(self.actual_values),
                'min_actual': np.min(self.actual_values),
                'max_actual': np.max(self.actual_values)
            }
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=4)
                
        return report
