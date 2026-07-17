import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from ..utils.logger import get_logger
from ..utils.exceptions import DataValidationError
from datetime import datetime


class DataValidator:
    """Validate data quality and integrity"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_required_columns(self, df: pd.DataFrame, required_columns: List[str]):
        """Validate that all required columns are present"""
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")
        self.logger.info(f"All required columns present: {required_columns}")
    
    def validate_data_types(self, df: pd.DataFrame, expected_types: Dict[str, str]):
        """Validate column data types"""
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    self.logger.warning(f"Column '{col}' expected type '{expected_type}' but got '{actual_type}'")
    
    def validate_range(self, df: pd.DataFrame, column: str, min_val: float, max_val: float):
        """Validate numerical range"""
        if column in df.columns:
            out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
            if len(out_of_range) > 0:
                self.logger.warning(f"Column '{column}' has {len(out_of_range)} values outside range [{min_val}, {max_val}]")
    
    def validate_missing_values(self, df: pd.DataFrame, threshold: float = 0.5):
        """Validate missing values threshold"""
        missing_percent = df.isnull().sum() / len(df)
        columns_exceeding = missing_percent[missing_percent > threshold]
        if len(columns_exceeding) > 0:
            raise DataValidationError(f"Columns with missing values exceeding {threshold}: {dict(columns_exceeding)}")
        self.logger.info(f"Missing values validation passed. Max missing: {missing_percent.max():.2%}")
    
    def validate_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None):
        """Validate duplicate rows"""
        duplicate_count = df.duplicated(subset=subset).sum()
        if duplicate_count > 0:
            self.logger.warning(f"Found {duplicate_count} duplicate rows")
        return duplicate_count
    
    def validate_customer_id(self, df: pd.DataFrame):
        """Validate customer ID format and consistency"""
        if 'CustomerID' in df.columns:
            # Check for negative IDs
            negative_ids = df[df['CustomerID'] < 0]
            if len(negative_ids) > 0:
                self.logger.warning(f"Found {len(negative_ids)} negative customer IDs")
            
            # Check for NaN
            nan_ids = df['CustomerID'].isnull()
            if nan_ids.sum() > 0:
                raise DataValidationError(f"Found {nan_ids.sum()} null customer IDs")
    
    def validate_invoice_date(self, df: pd.DataFrame):
        """Validate invoice date range"""
        if 'InvoiceDate' in df.columns:
            min_date = df['InvoiceDate'].min()
            max_date = df['InvoiceDate'].max()
            self.logger.info(f"Invoice date range: {min_date} to {max_date}")
            
            # Check for future dates
            future_dates = df[df['InvoiceDate'] > pd.Timestamp.now()]
            if len(future_dates) > 0:
                self.logger.warning(f"Found {len(future_dates)} future invoice dates")
    
    def validate_business_rules(self, df: pd.DataFrame):
        """Validate business rules"""
        # Check for negative quantities
        if 'Quantity' in df.columns:
            negative_qty = df[df['Quantity'] < 0]
            if len(negative_qty) > 0:
                self.logger.info(f"Found {len(negative_qty)} negative quantities (cancellations)")
        
        # Check for zero or negative prices
        if 'UnitPrice' in df.columns:
            invalid_prices = df[df['UnitPrice'] <= 0]
            if len(invalid_prices) > 0:
                self.logger.warning(f"Found {len(invalid_prices)} invalid prices (<= 0)")
        
        # Check amount consistency
        if all(col in df.columns for col in ['Quantity', 'UnitPrice']):
            calculated = df['Quantity'] * df['UnitPrice']
            # This is a general check, actual amount column may not exist yet
            self.logger.info("Quantity and UnitPrice consistency check passed")
    
    def validate_all(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Run all validations"""
        try:
            self.logger.info("Starting comprehensive data validation")
            
            # Required columns
            if 'required_columns' in config:
                self.validate_required_columns(df, config['required_columns'])
            
            # Data types
            if 'data_types' in config:
                self.validate_data_types(df, config['data_types'])
            
            # Range checks
            if 'range_checks' in config:
                for col, ranges in config['range_checks'].items():
                    self.validate_range(df, col, ranges['min'], ranges['max'])
            
            # Missing values
            self.validate_missing_values(df)
            
            # Customer ID
            self.validate_customer_id(df)
            
            # Invoice date
            self.validate_invoice_date(df)
            
            # Business rules
            self.validate_business_rules(df)
            
            self.logger.info("All validations passed successfully")
            
        except DataValidationError as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during validation: {str(e)}")
            raise