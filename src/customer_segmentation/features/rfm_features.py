import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
from ..utils.logger import get_logger
from ..utils.config import get_config


class RFMFeatureGenerator:
    """Generate RFM (Recency, Frequency, Monetary) features"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.segmentation_config = self.config.get_config('segmentation')
    
    def calculate_recency(self, df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.Series:
        """Calculate recency for each customer"""
        recency = df.groupby('CustomerID')['InvoiceDate'].max()
        recency = (reference_date - recency).dt.days
        self.logger.info(f"Recency calculated. Range: {recency.min()} to {recency.max()} days")
        return recency
    
    def calculate_frequency(self, df: pd.DataFrame) -> pd.Series:
        """Calculate frequency (number of transactions) for each customer"""
        frequency = df.groupby('CustomerID')['InvoiceNo'].nunique()
        self.logger.info(f"Frequency calculated. Range: {frequency.min()} to {frequency.max()} orders")
        return frequency
    
    def calculate_monetary(self, df: pd.DataFrame) -> pd.Series:
        """Calculate monetary value for each customer"""
        monetary = df.groupby('CustomerID')['Amount'].sum()
        self.logger.info(f"Monetary calculated. Range: {monetary.min():.2f} to {monetary.max():.2f}")
        return monetary
    
    def calculate_rfm_scores(self, df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate RFM scores and create RFM segment"""
        # Calculate base RFM values
        recency = self.calculate_recency(df, reference_date)
        frequency = self.calculate_frequency(df)
        monetary = self.calculate_monetary(df)
        
        # Create RFM DataFrame
        rfm = pd.DataFrame({
            'CustomerID': recency.index,
            'Recency': recency.values,
            'Frequency': frequency.values,
            'Monetary': monetary.values
        })
        
        # Calculate RFM scores (1-5 scale)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        # Calculate RFM score
        rfm['RFM_Score'] = rfm['R_Score'] * 100 + rfm['F_Score'] * 10 + rfm['M_Score']
        
        # Create RFM segment
        rfm['RFM_Segment'] = rfm.apply(self.get_rfm_segment, axis=1)
        
        self.logger.info(f"RFM scores calculated for {len(rfm)} customers")
        return rfm
    
    def get_rfm_segment(self, row: pd.Series) -> str:
        """Get RFM segment based on scores"""
        r_score = row['R_Score']
        f_score = row['F_Score']
        m_score = row['M_Score']
        
        if r_score >= 4 and f_score >= 4 and m_score >= 4:
            return 'Champions'
        elif r_score >= 3 and f_score >= 3 and m_score >= 3:
            return 'Loyal'
        elif r_score >= 4 and f_score >= 1 and m_score >= 1:
            return 'Potential'
        elif r_score <= 2 and f_score >= 4 and m_score >= 4:
            return 'At Risk'
        elif r_score <= 2 and f_score >= 2 and m_score >= 2:
            return 'Needs Attention'
        else:
            return 'Dormant'
    
    def get_customer_tier(self, rfm_row: pd.Series) -> str:
        """Get customer tier based on RFM scores"""
        monetary = rfm_row['Monetary']
        frequency = rfm_row['Frequency']
        recency = rfm_row['Recency']
        
        if monetary > 5000 and frequency > 10 and recency < 30:
            return 'Platinum'
        elif monetary > 2000 and frequency > 5 and recency < 60:
            return 'Gold'
        elif monetary > 500 and frequency > 3 and recency < 90:
            return 'Silver'
        else:
            return 'Bronze'
    
    def calculate_frequency_monetary_ratio(self, rfm: pd.DataFrame) -> pd.Series:
        """Calculate frequency to monetary ratio"""
        ratio = rfm['Monetary'] / (rfm['Frequency'] + 1)  # Add 1 to avoid division by zero
        return ratio
    
    def create_rfm_features(self, df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
        """Create complete RFM feature set"""
        # Calculate base RFM
        rfm = self.calculate_rfm_scores(df, reference_date)
        
        # Add derived features
        rfm['Frequency_Monetary_Ratio'] = self.calculate_frequency_monetary_ratio(rfm)
        rfm['Avg_Order_Value'] = rfm['Monetary'] / rfm['Frequency']
        rfm['Customer_Tier'] = rfm.apply(self.get_customer_tier, axis=1)
        
        # Add business rules
        rfm['Is_Champion'] = (rfm['RFM_Segment'] == 'Champions').astype(int)
        rfm['Is_At_Risk'] = (rfm['RFM_Segment'] == 'At Risk').astype(int)
        rfm['Is_Dormant'] = (rfm['RFM_Segment'] == 'Dormant').astype(int)
        
        return rfm