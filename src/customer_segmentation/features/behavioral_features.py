import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from ..utils.logger import get_logger
from ..utils.config import get_config


class BehavioralFeatureGenerator:
    """Generate behavioral features for customers"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
    
    def calculate_customer_lifetime(self, df: pd.DataFrame) -> pd.Series:
        """Calculate customer lifetime in days"""
        customer_lifetimes = df.groupby('CustomerID')['InvoiceDate'].agg(lambda x: (x.max() - x.min()).days)
        return customer_lifetimes
    
    def calculate_purchase_frequency(self, df: pd.DataFrame) -> pd.Series:
        """Calculate purchase frequency per month"""
        # Calculate days between purchases
        customer_purchases = df.groupby('CustomerID')['InvoiceDate'].agg(lambda x: x.tolist())
        frequencies = []
        
        for customer, dates in customer_purchases.items():
            if len(dates) > 1:
                sorted_dates = sorted(dates)
                intervals = [(sorted_dates[i+1] - sorted_dates[i]).days for i in range(len(sorted_dates)-1)]
                avg_interval = np.mean(intervals)
                # Convert to purchases per month
                frequency = 30 / avg_interval if avg_interval > 0 else 0
            else:
                frequency = 0
            frequencies.append((customer, frequency))
        
        return pd.Series({cust: freq for cust, freq in frequencies})
    
    def calculate_repeat_rate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate repeat purchase rate for each customer"""
        customer_orders = df.groupby('CustomerID')['InvoiceNo'].nunique()
        repeat_rate = (customer_orders > 1).astype(float)
        return repeat_rate
    
    def calculate_churn_risk(self, rfm_df: pd.DataFrame, threshold_days: int = 90) -> pd.Series:
        """Calculate churn risk based on recency"""
        churn_risk = rfm_df['Recency'].apply(
            lambda x: 1.0 if x > threshold_days else x / threshold_days
        )
        return churn_risk
    
    def calculate_engagement_score(self, rfm_df: pd.DataFrame) -> pd.Series:
        """Calculate engagement score based on multiple factors"""
        # Normalize RFM values
        recency_norm = 1 - (rfm_df['Recency'] / rfm_df['Recency'].max())
        frequency_norm = rfm_df['Frequency'] / rfm_df['Frequency'].max()
        monetary_norm = rfm_df['Monetary'] / rfm_df['Monetary'].max()
        
        # Weighted engagement score
        engagement = (recency_norm * 0.3 + frequency_norm * 0.3 + monetary_norm * 0.4)
        return engagement
    
    def calculate_seasonal_pattern(self, df: pd.DataFrame, customer_id: int) -> Dict[str, float]:
        """Calculate seasonal purchase patterns for a customer"""
        customer_data = df[df['CustomerID'] == customer_id]
        
        if len(customer_data) == 0:
            return {}
        
        # Extract month from invoice date
        customer_data['Month'] = customer_data['InvoiceDate'].dt.month
        customer_data['DayOfWeek'] = customer_data['InvoiceDate'].dt.dayofweek
        
        # Calculate seasonal patterns
        monthly_pattern = customer_data.groupby('Month')['Amount'].sum().to_dict()
        weekly_pattern = customer_data.groupby('DayOfWeek')['Amount'].sum().to_dict()
        
        # Calculate favorite month and day
        favorite_month = max(monthly_pattern, key=monthly_pattern.get) if monthly_pattern else None
        favorite_day = max(weekly_pattern, key=weekly_pattern.get) if weekly_pattern else None
        
        return {
            'monthly_pattern': monthly_pattern,
            'weekly_pattern': weekly_pattern,
            'favorite_month': favorite_month,
            'favorite_day': favorite_day,
            'purchase_count': len(customer_data)
        }
    
    def create_behavioral_features(self, df: pd.DataFrame, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive behavioral features"""
        self.logger.info("Creating behavioral features")
        
        # Merge with RFM data
        behavioral = rfm_df.copy()
        
        # Calculate additional features
        behavioral['Customer_Lifetime'] = self.calculate_customer_lifetime(df)
        behavioral['Purchase_Frequency'] = self.calculate_purchase_frequency(df)
        behavioral['Repeat_Rate'] = self.calculate_repeat_rate(df)
        behavioral['Churn_Risk'] = self.calculate_churn_risk(behavioral)
        behavioral['Engagement_Score'] = self.calculate_engagement_score(behavioral)
        
        # Calculate days since first purchase
        first_purchase = df.groupby('CustomerID')['InvoiceDate'].min()
        max_date = df['InvoiceDate'].max()
        behavioral['Days_Since_First'] = (max_date - first_purchase).dt.days
        
        self.logger.info(f"Created behavioral features for {len(behavioral)} customers")
        return behavioral