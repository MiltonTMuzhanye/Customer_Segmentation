import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.exceptions import DataValidationError


class DataIngestion:
    """Handle data ingestion from various sources"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.data_config = self.config.get_config('data')
    
    def load_excel(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Load Excel file"""
        try:
            self.logger.info(f"Loading Excel file: {file_path}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            self.logger.error(f"Error loading Excel file: {str(e)}")
            raise DataValidationError(f"Failed to load Excel: {str(e)}")
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load CSV file"""
        try:
            self.logger.info(f"Loading CSV file: {file_path}")
            df = pd.read_csv(file_path, **kwargs)
            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {str(e)}")
            raise DataValidationError(f"Failed to load CSV: {str(e)}")
    
    def load_database(self, query: str, connection_string: str) -> pd.DataFrame:
        """Load data from database"""
        try:
            import sqlalchemy
            self.logger.info(f"Loading data from database")
            engine = sqlalchemy.create_engine(connection_string)
            df = pd.read_sql(query, engine)
            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            self.logger.error(f"Error loading from database: {str(e)}")
            raise DataValidationError(f"Failed to load from database: {str(e)}")
    
    def load_from_api(self, endpoint: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Load data from API"""
        try:
            import requests
            self.logger.info(f"Loading data from API: {endpoint}")
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            self.logger.error(f"Error loading from API: {str(e)}")
            raise DataValidationError(f"Failed to load from API: {str(e)}")
    
    def save_data(self, df: pd.DataFrame, file_path: str, format: str = 'csv'):
        """Save data to file"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'csv':
                df.to_csv(file_path, index=False)
            elif format == 'excel':
                df.to_excel(file_path, index=False)
            elif format == 'parquet':
                df.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved data to: {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise