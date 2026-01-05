"""
Data cleaning module for customer segmentation system
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Handles data cleaning and preprocessing"""
    
    def __init__(self):
        """Initialize DataCleaner"""
        self.processed_dir = Path('../data/processed/')
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_retail_data(self, df):
        """
        Clean the online retail dataset
        
        Args:
            df: Raw pandas DataFrame
            
        Returns:
            Cleaned pandas DataFrame
        """
        logger.info("Starting data cleaning process...")
        
        # Create a copy to avoid modifying original
        data = df.copy()
        original_shape = data.shape
        logger.info(f"Original data shape: {original_shape}")
        
        # 1. Remove rows with missing CustomerID
        missing_customers_before = data['CustomerID'].isna().sum()
        data = data[data['CustomerID'].notna()]
        missing_customers_after = data['CustomerID'].isna().sum()
        logger.info(f"Removed {missing_customers_before - missing_customers_after} rows with missing CustomerID")
        
        # 2. Convert CustomerID to integer
        data['CustomerID'] = data['CustomerID'].astype(int)
        logger.info("Converted CustomerID to integer")
        
        # 3. Remove cancellations (negative quantities)
        cancellations_before = len(data[data['Quantity'] < 0])
        data = data[data['Quantity'] > 0]
        cancellations_after = len(data[data['Quantity'] < 0])
        logger.info(f"Removed {cancellations_before - cancellations_after} cancellation transactions")
        
        # 4. Calculate transaction amount
        data['Amount'] = data['Quantity'] * data['UnitPrice']
        logger.info("Calculated transaction amount")
        
        # 5. Add date features
        data['YearMonth'] = data['InvoiceDate'].dt.to_period('M')
        data['DayOfWeek'] = data['InvoiceDate'].dt.day_name()
        data['Hour'] = data['InvoiceDate'].dt.hour
        logger.info("Added date features")
        
        # 6. Check for outliers in Amount
        amount_stats = data['Amount'].describe()
        q1, q3 = amount_stats['25%'], amount_stats['75%']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = data[(data['Amount'] < lower_bound) | (data['Amount'] > upper_bound)]
        logger.info(f"Found {len(outliers)} amount outliers using IQR method")
        
        # 7. Validate data
        validation_result = self._validate_data(data)
        for check, result in validation_result.items():
            if result['status'] == 'FAIL':
                logger.warning(f"Validation check failed: {check} - {result['message']}")
            else:
                logger.info(f"Validation check passed: {check}")
        
        logger.info(f"Cleaned data shape: {data.shape}")
        logger.info(f"Removed {original_shape[0] - data.shape[0]} rows total")
        
        return data
    
    def _validate_data(self, df):
        """
        Validate cleaned data
        
        Args:
            df: Cleaned pandas DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        # Check 1: No missing CustomerID
        missing_customers = df['CustomerID'].isna().sum()
        validation_results['missing_customers'] = {
            'status': 'PASS' if missing_customers == 0 else 'FAIL',
            'message': f'{missing_customers} missing CustomerIDs'
        }
        
        # Check 2: All quantities positive
        negative_quantities = len(df[df['Quantity'] <= 0])
        validation_results['negative_quantities'] = {
            'status': 'PASS' if negative_quantities == 0 else 'FAIL',
            'message': f'{negative_quantities} non-positive quantities'
        }
        
        # Check 3: Valid dates
        invalid_dates = df['InvoiceDate'].isna().sum()
        validation_results['invalid_dates'] = {
            'status': 'PASS' if invalid_dates == 0 else 'FAIL',
            'message': f'{invalid_dates} invalid dates'
        }
        
        # Check 4: Valid amounts
        invalid_amounts = len(df[df['Amount'] <= 0])
        validation_results['invalid_amounts'] = {
            'status': 'PASS' if invalid_amounts == 0 else 'FAIL',
            'message': f'{invalid_amounts} invalid amounts'
        }
        
        return validation_results
    
    def save_cleaned_data(self, df, file_name='cleaned_customers.csv'):
        """
        Save cleaned data to CSV
        
        Args:
            df: Cleaned pandas DataFrame
            file_name: Output file name
        """
        output_path = self.processed_dir / file_name
        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to {output_path} ({len(df)} rows)")

# Example usage
if __name__ == "__main__":
    from load_data import DataLoader
    
    # Load and clean data
    loader = DataLoader()
    cleaner = DataCleaner()
    
    # Load raw data
    raw_data = loader.load_retail_data()
    
    # Clean data
    cleaned_data = cleaner.clean_retail_data(raw_data)
    
    # Save cleaned data
    cleaner.save_cleaned_data(cleaned_data)
    
    print(f"\nCleaning complete!")
    print(f"Original rows: {len(raw_data)}")
    print(f"Cleaned rows: {len(cleaned_data)}")
    print(f"Retention rate: {len(cleaned_data)/len(raw_data)*100:.1f}%")