from datetime import datetime
import schedule
import time
from .data_connector import PowerBIConnector

class PowerBIScheduler:
    def __init__(self, connector: PowerBIConnector):
        """Initialize scheduler with PowerBI connector"""
        self.connector = connector

    def schedule_refresh(self, interval_hours=24):
        """
        Schedule regular data refreshes
        
        Args:
            interval_hours (int): Refresh interval in hours
        """
        def refresh_job():
            print(f"Starting scheduled refresh at {datetime.now()}")
            # Add your refresh logic here
            
        # Schedule job
        schedule.every(interval_hours).hours.do(refresh_job)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
