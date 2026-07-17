import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from scipy import stats
from ..utils.logger import get_logger
from ..utils.config import get_config


class DataPreprocessor:
    """Preprocess data for segmentation"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.segmentation_config = self.config.get_config('segmentation')
        self.data_config = self.config.get_config('data')
    
    def remove_cancellations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove cancelled transactions"""
        before = len(df)
        df = df[df['Quantity'] > 0]
        removed = before - len(df)
        self.logger.info(f"Removed {removed} cancelled transactions")
        return df
    
    def remove_missing_customer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove transactions without customer ID"""
        before = len(df)
        df = df[df['CustomerID'].notna()]
        removed = before - len(df)
        self.logger.info(f"Removed {removed} transactions with missing customer ID")
        return df
    
    def filter_quantity(self, df: pd.DataFrame, min_quantity: int = 1) -> pd.DataFrame:
        """Filter by minimum quantity"""
        before = len(df)
        df = df[df['Quantity'] >= min_quantity]
        removed = before - len(df)
        self.logger.info(f"Removed {removed} transactions with quantity < {min_quantity}")
        return df
    
    def remove_outliers_iqr(self, df: pd.DataFrame, columns: List[str], threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        before = len(df)
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        removed = before - len(df)
        self.logger.info(f"Removed {removed} outlier rows using IQR method")
        return df
    
    def remove_outliers_zscore(self, df: pd.DataFrame, columns: List[str], threshold: float = 3) -> pd.DataFrame:
        """Remove outliers using Z-score method"""
        before = len(df)
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                df = df[~((z_scores > threshold) & (df[col].notna()))]
        
        removed = before - len(df)
        self.logger.info(f"Removed {removed} outlier rows using Z-score method")
        return df
    
    def create_amount_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount column from quantity and unit price"""
        if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
            df['Amount'] = df['Quantity'] * df['UnitPrice']
            self.logger.info(f"Created 'Amount' column. Range: {df['Amount'].min():.2f} to {df['Amount'].max():.2f}")
        return df
    
    def convert_customer_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert customer ID to int"""
        if 'CustomerID' in df.columns:
            df['CustomerID'] = df['CustomerID'].astype(int)
            self.logger.info(f"Converted CustomerID to int. Unique values: {df['CustomerID'].nunique()}")
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute full preprocessing pipeline"""
        self.logger.info("Starting full preprocessing pipeline")
        
        # Get configuration
        preprocess_config = self.data_config.get('preprocessing', {})
        
        # Clean customer data
        if preprocess_config.get('remove_missing_customer', True):
            df = self.remove_missing_customer(df)
        
        # Convert customer ID
        df = self.convert_customer_id(df)
        
        # Remove cancellations
        if preprocess_config.get('remove_cancellations', True):
            df = self.remove_cancellations(df)
        
        # Filter by quantity
        min_qty = preprocess_config.get('min_quantity', 1)
        df = self.filter_quantity(df, min_qty)
        
        # Create amount
        df = self.create_amount_column(df)
        
        # Remove outliers
        outlier_method = preprocess_config.get('outlier_method', 'iqr')
        if outlier_method == 'iqr':
            df = self.remove_outliers_iqr(df, ['Quantity', 'UnitPrice', 'Amount'])
        elif outlier_method == 'zscore':
            df = self.remove_outliers_zscore(df, ['Quantity', 'UnitPrice', 'Amount'])
        
        self.logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df
    
    def prepare_for_rfm(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
        """Prepare data for RFM analysis"""
        # Ensure InvoiceDate is datetime
        if 'InvoiceDate' in df.columns:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            reference_date = df['InvoiceDate'].max() + timedelta(days=1)
            self.logger.info(f"Reference date for recency: {reference_date}")
        else:
            reference_date = pd.Timestamp.now()
        
        return df, reference_date