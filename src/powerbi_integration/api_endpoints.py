from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from .data_connector import PowerBIConnector

app = FastAPI()
connector = PowerBIConnector()

class PredictionData(BaseModel):
    date: datetime
    predicted_profit: float
    actual_profit: float
    features: dict

@app.post("/update_predictions")
async def update_predictions(data: PredictionData):
    """
    API endpoint to update prediction data
    
    Args:
        data (PredictionData): New prediction data
    Returns:
        dict: Status message
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame([{
            'Date': data.date,
            'PredictedProfit': data.predicted_profit,
            'ActualProfit': data.actual_profit,
            **data.features
        }])
        
        # Upload to Azure
        connector.upload_to_azure(df, f'predictions_{data.date.date()}.csv')
        
        return {"status": "success", "message": "Data updated successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
