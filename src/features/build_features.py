"""
Feature engineering module for customer segmentation system
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureBuilder:
    """Builds RFM and behavioral features for customer segmentation"""
    
    def __init__(self, reference_date=None):
        """
        Initialize FeatureBuilder
        
        Args:
            reference_date: Reference date for recency calculation.
                          If None, will use max date in data + 1 day
        """
        self.reference_date = reference_date
        self.scaler = StandardScaler()
        
        # Create directories if they don't exist
        self.features_dir = Path('../data/features/')
        self.models_dir = Path('../models/')
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def build_rfm_features(self, df):
        """
        Build RFM (Recency, Frequency, Monetary) features
        
        Args:
            df: Cleaned transaction data with CustomerID, InvoiceDate, Amount
            
        Returns:
            DataFrame with RFM features for each customer
        """
        logger.info("Building RFM features...")
        
        if self.reference_date is None:
            self.reference_date = df['InvoiceDate'].max() + timedelta(days=1)
            logger.info(f"Using reference date: {self.reference_date}")
        
        # Group by customer
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (self.reference_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'Amount': 'sum'
        }).rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency',
            'Amount': 'Monetary'
        })
        
        logger.info(f"Created RFM features for {len(rfm)} customers")
        return rfm
    
    def build_behavioral_features(self, df, rfm_df):
        """
        Build additional behavioral features
        
        Args:
            df: Cleaned transaction data
            rfm_df: DataFrame with basic RFM features
            
        Returns:
            DataFrame with comprehensive behavioral features
        """
        logger.info("Building behavioral features...")
        
        # Aggregate customer-level statistics
        customer_stats = df.groupby('CustomerID').agg({
            'Amount': ['mean', 'std', 'count'],
            'Quantity': 'sum',
            'InvoiceDate': ['min', 'max', 'nunique'],
            'StockCode': 'nunique'
        })
        
        # Flatten column names
        customer_stats.columns = ['_'.join(col).strip() for col in customer_stats.columns.values]
        customer_stats = customer_stats.rename(columns={
            'Amount_mean': 'AvgTransactionValue',
            'Amount_std': 'StdTransactionValue',
            'Amount_count': 'TotalTransactions',
            'Quantity_sum': 'TotalQuantity',
            'InvoiceDate_min': 'FirstPurchaseDate',
            'InvoiceDate_max': 'LastPurchaseDate',
            'InvoiceDate_nunique': 'UniquePurchaseDays',
            'StockCode_nunique': 'UniqueProducts'
        })
        
        # Calculate derived metrics
        customer_stats['CustomerLifetime'] = (customer_stats['LastPurchaseDate'] - 
                                              customer_stats['FirstPurchaseDate']).dt.days + 1
        customer_stats['PurchaseFrequency'] = customer_stats['TotalTransactions'] / customer_stats['UniquePurchaseDays']
        customer_stats['AvgBasketSize'] = customer_stats['TotalQuantity'] / customer_stats['TotalTransactions']
        customer_stats['ProductVariety'] = customer_stats['UniqueProducts'] / customer_stats['TotalTransactions']
        
        # Merge with RFM
        full_features = rfm_df.merge(customer_stats, left_index=True, right_index=True)
        
        logger.info(f"Created {full_features.shape[1]} features for {full_features.shape[0]} customers")
        return full_features
    
    def prepare_for_clustering(self, features_df):
        """
        Prepare features for clustering algorithm
        
        Args:
            features_df: DataFrame with all features
            
        Returns:
            Tuple of (scaled_features_df, feature_names)
        """
        logger.info("Preparing features for clustering...")
        
        # Select features for clustering
        clustering_features = [
            'Recency',
            'Frequency',
            'Monetary',
            'AvgTransactionValue',
            'CustomerLifetime',
            'PurchaseFrequency'
        ]
        
        # Filter out features that might not exist
        available_features = [f for f in clustering_features if f in features_df.columns]
        logger.info(f"Using {len(available_features)} features for clustering: {available_features}")
        
        # Filter out customers with $0 spent
        features_filtered = features_df[features_df['Monetary'] > 0].copy()
        zero_spend = len(features_df) - len(features_filtered)
        if zero_spend > 0:
            logger.info(f"Filtered out {zero_spend} customers with $0 total spend")
        
        # Handle missing values
        features_filtered = features_filtered[available_features].copy()
        missing_before = features_filtered.isnull().sum().sum()
        features_filtered = features_filtered.fillna(features_filtered.median())
        missing_after = features_filtered.isnull().sum().sum()
        
        if missing_before > 0:
            logger.info(f"Filled {missing_before - missing_after} missing values with median")
        
        # Apply log transformation to skewed features
        skewed_features = ['Frequency', 'Monetary', 'AvgTransactionValue', 'TotalTransactions']
        
        for feature in skewed_features:
            if feature in features_filtered.columns:
                features_filtered[f'Log_{feature}'] = np.log1p(features_filtered[feature])
        
        # Update feature list with log-transformed versions
        final_features = []
        for feature in available_features:
            if feature in skewed_features and f'Log_{feature}' in features_filtered.columns:
                final_features.append(f'Log_{feature}')
            elif feature in features_filtered.columns:
                final_features.append(feature)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features_filtered[final_features])
        scaled_df = pd.DataFrame(scaled_features, 
                                 columns=final_features, 
                                 index=features_filtered.index)
        
        logger.info(f"Final feature matrix shape: {scaled_df.shape}")
        return scaled_df, final_features
    
    def save_features(self, scaled_df, original_df, feature_names):
        """
        Save features for modeling
        
        Args:
            scaled_df: Scaled feature matrix
            original_df: Original feature DataFrame
            feature_names: List of feature names used
        """
        # Save scaled features
        scaled_path = self.features_dir / 'customer_features.parquet'
        scaled_df.to_parquet(scaled_path)
        logger.info(f"Saved scaled features to {scaled_path}")
        
        # Save original features
        original_path = self.features_dir / 'customer_rfm_full.parquet'
        original_df.to_parquet(original_path)
        logger.info(f"Saved original features to {original_path}")
        
        # Save scaler
        scaler_path = self.models_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save feature metadata
        metadata = {
            'feature_names': feature_names,
            'n_customers': len(scaled_df),
            'n_features': len(feature_names),
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
        }
        
        import json
        metadata_path = self.models_dir / 'feature_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved feature metadata to {metadata_path}")

# Example usage
if __name__ == "__main__":
    from src.data.load_data import DataLoader
    from src.data.clean_data import DataCleaner
    
    # Load and clean data
    loader = DataLoader()
    cleaner = DataCleaner()
    feature_builder = FeatureBuilder()
    
    # Load raw data
    raw_data = loader.load_retail_data()
    
    # Clean data
    cleaned_data = cleaner.clean_retail_data(raw_data)
    
    # Build features
    rfm_features = feature_builder.build_rfm_features(cleaned_data)
    all_features = feature_builder.build_behavioral_features(cleaned_data, rfm_features)
    
    # Prepare for clustering
    scaled_features, feature_names = feature_builder.prepare_for_clustering(all_features)
    
    # Save features
    feature_builder.save_features(scaled_features, all_features, feature_names)
    
    print(f"\nFeature engineering complete!")
    print(f"Total customers: {len(scaled_features)}")
    print(f"Features used: {feature_names}")