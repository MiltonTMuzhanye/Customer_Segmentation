"""
Tests for data loading and cleaning modules
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.load_data import DataLoader
from src.data.clean_data import DataCleaner

@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing"""
    dates = [datetime(2023, 1, i) for i in range(1, 6)]
    
    data = pd.DataFrame({
        'InvoiceNo': [f'INV{i:03d}' for i in range(10)],
        'StockCode': [f'CODE{i}' for i in range(10)],
        'Description': [f'Product {i}' for i in range(10)],
        'Quantity': [1, 2, 3, 4, 5, -1, -2, 1, 2, 3],  # Includes negative quantities
        'InvoiceDate': dates * 2,
        'UnitPrice': [10.0, 15.0, 20.0, 25.0, 30.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        'CustomerID': [1.0, 1.0, 1.0, 2.0, 2.0, np.nan, 3.0, 3.0, 3.0, 4.0],  # Includes NaN
        'Country': ['UK', 'UK', 'UK', 'US', 'US', 'UK', 'UK', 'UK', 'US', 'US']
    })
    
    return data

@pytest.fixture
def data_loader(tmp_path):
    """Create DataLoader with temporary directory"""
    # Create temporary data directory
    data_dir = tmp_path / 'raw'
    data_dir.mkdir(parents=True)
    
    return DataLoader(data_dir=str(data_dir))

@pytest.fixture
def data_cleaner(tmp_path):
    """Create DataCleaner with temporary directory"""
    return DataCleaner()

class TestDataLoader:
    """Test DataLoader class"""
    
    def test_load_retail_data(self, data_loader, sample_raw_data, tmp_path):
        """Test loading retail data from Excel file"""
        # Save sample data to temporary Excel file
        data_path = tmp_path / 'raw' / 'Online Retail.xlsx'
        sample_raw_data.to_excel(data_path, index=False)
        
        # Load the data
        loaded_data = data_loader.load_retail_data('Online Retail.xlsx')
        
        # Check data was loaded correctly
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_raw_data)
        assert all(col in loaded_data.columns for col in sample_raw_data.columns)
        
        # Check date range is calculated
        assert 'InvoiceDate' in loaded_data.columns
        assert loaded_data['InvoiceDate'].min() <= loaded_data['InvoiceDate'].max()
        
    def test_load_nonexistent_file(self, data_loader):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            data_loader.load_retail_data('nonexistent.xlsx')
            
    def test_load_processed_data(self, data_loader, sample_raw_data, tmp_path):
        """Test loading processed CSV data"""
        # Save sample data to temporary CSV file
        data_path = tmp_path / 'processed' / 'cleaned_customers.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert InvoiceDate to string for CSV
        sample_for_csv = sample_raw_data.copy()
        sample_for_csv['InvoiceDate'] = sample_for_csv['InvoiceDate'].astype(str)
        sample_for_csv.to_csv(data_path, index=False)
        
        # Load the data
        loaded_data = data_loader.load_processed_data('cleaned_customers.csv')
        
        # Check data was loaded correctly
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_for_csv)
        assert 'InvoiceDate' in loaded_data.columns
        
        # Check InvoiceDate was parsed as datetime
        assert pd.api.types.is_datetime64_any_dtype(loaded_data['InvoiceDate'])

class TestDataCleaner:
    """Test DataCleaner class"""
    
    def test_clean_retail_data(self, sample_raw_data, data_cleaner):
        """Test cleaning retail data"""
        cleaned_data = data_cleaner.clean_retail_data(sample_raw_data)
        
        # Check data type
        assert isinstance(cleaned_data, pd.DataFrame)
        
        # Check missing CustomerIDs were removed
        original_missing = sample_raw_data['CustomerID'].isna().sum()
        cleaned_missing = cleaned_data['CustomerID'].isna().sum()
        assert cleaned_missing == 0
        assert len(cleaned_data) == len(sample_raw_data) - original_missing
        
        # Check CustomerID converted to integer
        assert pd.api.types.is_integer_dtype(cleaned_data['CustomerID'])
        
        # Check negative quantities were removed
        original_negatives = (sample_raw_data['Quantity'] <= 0).sum()
        cleaned_negatives = (cleaned_data['Quantity'] <= 0).sum()
        assert cleaned_negatives == 0
        assert len(cleaned_data) == len(sample_raw_data) - original_missing - original_negatives
        
        # Check Amount column was added
        assert 'Amount' in cleaned_data.columns
        assert (cleaned_data['Amount'] == cleaned_data['Quantity'] * cleaned_data['UnitPrice']).all()
        
        # Check date features were added
        assert 'YearMonth' in cleaned_data.columns
        assert 'DayOfWeek' in cleaned_data.columns
        assert 'Hour' in cleaned_data.columns
        
    def test_clean_data_with_no_negatives(self):
        """Test cleaning data with no negative quantities"""
        data = pd.DataFrame({
            'CustomerID': [1.0, 2.0, 3.0],
            'Quantity': [1, 2, 3],
            'UnitPrice': [10.0, 20.0, 30.0],
            'InvoiceDate': [datetime(2023, 1, i) for i in range(1, 4)]
        })
        
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_retail_data(data)
        
        # Should keep all rows
        assert len(cleaned_data) == 3
        
    def test_clean_data_all_valid(self):
        """Test cleaning data with all valid entries"""
        data = pd.DataFrame({
            'CustomerID': [1.0, 2.0, 3.0],
            'Quantity': [1, 2, 3],
            'UnitPrice': [10.0, 20.0, 30.0],
            'InvoiceDate': [datetime(2023, 1, i) for i in range(1, 4)],
            'InvoiceNo': ['INV001', 'INV002', 'INV003'],
            'StockCode': ['A', 'B', 'C'],
            'Description': ['Prod A', 'Prod B', 'Prod C'],
            'Country': ['UK', 'US', 'UK']
        })
        
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_retail_data(data)
        
        # Should keep all rows
        assert len(cleaned_data) == 3
        assert 'Amount' in cleaned_data.columns
        
    def test_validate_data(self, data_cleaner):
        """Test data validation"""
        # Create valid data
        valid_data = pd.DataFrame({
            'CustomerID': [1, 2, 3],
            'Quantity': [1, 2, 3],
            'InvoiceDate': [datetime(2023, 1, i) for i in range(1, 4)],
            'Amount': [10.0, 20.0, 30.0]
        })
        
        validation_results = data_cleaner._validate_data(valid_data)
        
        # All checks should pass
        for check, result in validation_results.items():
            assert result['status'] == 'PASS', f"Check {check} failed: {result['message']}"
            
    def test_validate_data_with_issues(self, data_cleaner):
        """Test data validation with issues"""
        # Create data with issues
        invalid_data = pd.DataFrame({
            'CustomerID': [1, np.nan, 3],  # Missing customer
            'Quantity': [1, 0, -1],  # Non-positive quantities
            'InvoiceDate': [datetime(2023, 1, 1), None, datetime(2023, 1, 3)],  # Missing date
            'Amount': [10.0, 0.0, -5.0]  # Invalid amounts
        })
        
        validation_results = data_cleaner._validate_data(invalid_data)
        
        # Some checks should fail
        failed_checks = [check for check, result in validation_results.items() 
                        if result['status'] == 'FAIL']
        assert len(failed_checks) > 0
        
    def test_save_cleaned_data(self, sample_raw_data, data_cleaner, tmp_path):
        """Test saving cleaned data"""
        # Clean the data
        cleaned_data = data_cleaner.clean_retail_data(sample_raw_data)
        
        # Save to temporary location
        save_path = tmp_path / 'cleaned_test.csv'
        data_cleaner.save_cleaned_data(cleaned_data, file_name=str(save_path))
        
        # Check file was created
        assert save_path.exists()
        
        # Load and verify
        loaded_data = pd.read_csv(save_path)
        assert len(loaded_data) == len(cleaned_data)
        
    def test_empty_dataframe(self, data_cleaner):
        """Test cleaning empty dataframe"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):
            data_cleaner.clean_retail_data(empty_df)
            
    def test_data_without_required_columns(self, data_cleaner):
        """Test cleaning data without required columns"""
        incomplete_df = pd.DataFrame({
            'CustomerID': [1, 2, 3],
            # Missing InvoiceDate, Quantity, etc.
        })
        
        with pytest.raises(Exception):
            data_cleaner.clean_retail_data(incomplete_df)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])