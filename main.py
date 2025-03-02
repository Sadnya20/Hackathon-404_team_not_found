import argparse
import os
import json
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model_training import AirlineProfitModel
from src.utils import evaluate_model, plot_feature_importance
from src.input_output import AirlineProfitIO
from src.powerbi_integration.data_connector import PowerBIConnector
from src.powerbi_integration.refresh_scheduler import PowerBIScheduler
import threading

class AirlineProfitPrediction:
    def __init__(self):
        self.model = None
        self.io_handler = None
        self.powerbi_connector = None

    def train_model(self, data_path: str, model_save_path: str) -> None:
        """
        Train the airline profit prediction model
        
        Args:
            data_path (str): Path to input data
            model_save_path (str): Path to save trained model
        """
        try:
            # 1. Load and preprocess data
            print("Loading data...")
            df = load_data(data_path)
            if df is None:
                return
            
            print("\nPreprocessing data...")
            df_processed = preprocess_data(df)
            X, y = split_data(df_processed)
            
            # 2. Train model
            print("\nTraining model...")
            self.model = AirlineProfitModel(n_estimators=100)
            training_results = self.model.train(X, y)
            
            print("\nTraining Results:")
            print(f"Cross-validation mean score: {training_results['cv_scores_mean']:.4f}")
            print(f"Cross-validation std: {training_results['cv_scores_std']:.4f}")
            
            # 3. Evaluate model
            print("\nEvaluating model...")
            metrics = evaluate_model(self.model.model, self.model.X_test, self.model.y_test)
            print("\nModel Performance:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # 4. Save model
            if model_save_path:
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                self.model.save_model(model_save_path)
                print(f"\nModel saved to {model_save_path}")
            
            # 5. Plot feature importance
            print("\nGenerating feature importance plot...")
            plot_feature_importance(self.model.model, X)
            
            # 6. Initialize IO handler
            self.io_handler = AirlineProfitIO(self.model)
            
        except Exception as e:
            print(f"Error in training: {str(e)}")

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model
        
        Args:
            model_path (str): Path to saved model
        """
        try:
            self.model = AirlineProfitModel()
            self.model.load_model(model_path)
            self.io_handler = AirlineProfitIO(self.model)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")

    def process_prediction(self, input_type: str, input_data: str) -> None:
        """
        Process prediction request
        
        Args:
            input_type (str): 'single' or 'batch'
            input_data (str): Path to JSON input file or JSON string
        """
        if self.io_handler is None:
            print("Error: Model not loaded. Please train or load a model first.")
            return

        try:
            # Load input data
            if input_data.endswith('.json'):
                with open(input_data, 'r') as f:
                    data = json.load(f)
            else:
                data = json.loads(input_data)

            # Process input
            if input_type == 'single':
                result = self.io_handler.process_single_input(data)
            else:
                result = self.io_handler.process_batch_input(data)
            
            # Print results
            print("\nPrediction Results:")
            print(json.dumps(result, indent=2))

            # Update PowerBI if configured
            if self.powerbi_connector:
                self.powerbi_connector.upload_to_azure(
                    pd.DataFrame([result]),
                    f'predictions_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
                )

        except Exception as e:
            print(f"Error processing prediction: {str(e)}")

    def start_powerbi_integration(self) -> None:
        """Initialize and start PowerBI integration"""
        try:
            self.powerbi_connector = PowerBIConnector()
            scheduler = PowerBIScheduler(self.powerbi_connector)
            
            # Start scheduler in separate thread
            scheduler_thread = threading.Thread(
                target=scheduler.schedule_refresh,
                args=(24,)  # 24-hour refresh interval
            )
            scheduler_thread.daemon = True
            scheduler_thread.start()
            print("PowerBI integration started successfully")
        except Exception as e:
            print(f"Error starting PowerBI integration: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Airline Profit Prediction System")
    parser.add_argument(
        "--mode",
        choices=['train', 'predict'],
        default='predict',
        help="Mode of operation"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/airline_data.csv",
        help="Path to input data CSV file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/saved_models/model.joblib",
        help="Path to save/load model"
    )
    parser.add_argument(
        "--input_type",
        choices=['single', 'batch'],
        default='single',
        help="Type of input prediction"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSON string or file path"
    )
    parser.add_argument(
        "--powerbi",
        action='store_true',
        help="Enable PowerBI integration"
    )

    args = parser.parse_args()
    
    # Initialize system
    system = AirlineProfitPrediction()

    # Enable PowerBI integration if requested
    if args.powerbi:
        system.start_powerbi_integration()

    if args.mode == 'train':
        system.train_model(args.data_path, args.model_path)
    else:
        # Load model for prediction
        system.load_model(args.model_path)
        if args.input:
            system.process_prediction(args.input_type, args.input)
        else:
            print("Error: Input data required for prediction mode")

if __name__ == "__main__":
    main()
