import pandas as pd
from datetime import datetime
import pyodbc
from azure.storage.blob import BlobServiceClient
import json

class PowerBIConnector:
    def __init__(self, config_path='config/powerbi_config.json'):
        """Initialize PowerBI connector with configuration"""
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.connection_string = self.config['connection_string']
        self.container_name = self.config['container_name']
        self.workspace_id = self.config['workspace_id']

    def prepare_prediction_data(self, predictions_df, actual_df):
        """
        Prepare data for PowerBI
        
        Args:
            predictions_df (pd.DataFrame): Model predictions
            actual_df (pd.DataFrame): Actual values
        Returns:
            pd.DataFrame: Formatted data for PowerBI
        """
        # Merge predictions with actual values
        combined_df = pd.merge(
            predictions_df,
            actual_df,
            on='Date',
            suffixes=('_predicted', '_actual')
        )
        
        # Add timestamp
        combined_df['LastUpdated'] = datetime.now()
        
        return combined_df

    def upload_to_azure(self, df, blob_name):
        """
        Upload data to Azure Blob Storage for PowerBI
        
        Args:
            df (pd.DataFrame): Data to upload
            blob_name (str): Name of the blob
        """
        try:
            # Create blob service client
            blob_service_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            
            # Get container client
            container_client = blob_service_client.get_container_client(
                self.container_name
            )
            
            # Convert DataFrame to CSV
            csv_data = df.to_csv(index=False)
            
            # Upload to blob
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(csv_data, overwrite=True)
            
            print(f"Successfully uploaded data to {blob_name}")
            
        except Exception as e:
            print(f"Error uploading to Azure: {str(e)}")

    def refresh_dataset(self, dataset_id):
        """
        Trigger PowerBI dataset refresh
        
        Args:
            dataset_id (str): PowerBI dataset ID
        """
        try:
            # Implementation depends on PowerBI API authentication method
            pass
        except Exception as e:
            print(f"Error refreshing dataset: {str(e)}")
