# src/input_output.py

import pandas as pd
import numpy as np
from typing import Dict, Union, List
import json

class AirlineProfitIO:
    def __init__(self, model):
        """
        Initialize with trained model
        Args:
            model: Trained machine learning model
        """
        self.model = model

    def process_single_input(self, input_data: Dict) -> Dict:
        """
        Process single prediction input
        
        Args:
            input_data (dict): Dictionary containing input features
            
        Returns:
            dict: Prediction results
        """
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            
            return {
                'status': 'success',
                'predicted_profit': float(prediction),
                'input_data': input_data
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def process_batch_input(self, input_data: List[Dict]) -> Dict:
        """
        Process batch predictions
        
        Args:
            input_data (list): List of dictionaries containing input features
            
        Returns:
            dict: Batch prediction results
        """
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame(input_data)
            
            # Make predictions
            predictions = self.model.predict(input_df)
            
            return {
                'status': 'success',
                'predictions': predictions.tolist(),
                'input_data': input_data
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
