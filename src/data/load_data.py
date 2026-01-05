"""
Data loading module for customer segmentation system
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading of raw customer data"""
    
    def __init__(self, data_dir='../data/raw/'):
        """
        Initialize DataLoader
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        
    def load_retail_data(self, file_name='Online Retail.xlsx'):
        """
        Load online retail dataset
        
        Args:
            file_name: Name of the Excel file
            
        Returns:
            pandas DataFrame with the loaded data
        """
        try:
            file_path = self.data_dir / file_name
            logger.info(f"Loading data from {file_path}")
            
            df = pd.read_excel(file_path)
            logger.info(f"Successfully loaded {len(df)} rows, {df.shape[1]} columns")
            
            # Basic info
            logger.info(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
            logger.info(f"Unique customers: {df['CustomerID'].nunique()}")
            logger.info(f"Unique products: {df['StockCode'].nunique()}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def load_processed_data(self, file_name='cleaned_customers.csv'):
        """
        Load processed data
        
        Args:
            file_name: Name of the processed CSV file
            
        Returns:
            pandas DataFrame with the loaded data
        """
        try:
            file_path = Path('../data/processed/') / file_name
            logger.info(f"Loading processed data from {file_path}")
            
            df = pd.read_csv(file_path, parse_dates=['InvoiceDate'])
            logger.info(f"Successfully loaded {len(df)} rows")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    
    # Load raw data
    raw_data = loader.load_retail_data()
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Columns: {raw_data.columns.tolist()}")
    
    # Load processed data
    try:
        processed_data = loader.load_processed_data()
        print(f"\nProcessed data shape: {processed_data.shape}")
    except FileNotFoundError:
        print("\nProcessed data not found - run cleaning first")