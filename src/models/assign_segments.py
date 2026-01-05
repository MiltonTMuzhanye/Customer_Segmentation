"""
Segment assignment module for customer segmentation system
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentAssigner:
    """Assigns new customers to segments using trained model"""
    
    def __init__(self):
        """Initialize SegmentAssigner"""
        self.models_dir = Path('../models/')
        
        # Load trained components
        self.model = None
        self.scaler = None
        self.pca_2d = None
        self.pca_3d = None
        self.feature_names = None
        self.segment_names = None
        
        self._load_components()
    
    def _load_components(self):
        """Load all necessary components"""
        try:
            # Load model
            model_path = self.models_dir / 'segmentation_model.pkl'
            self.model = joblib.load(model_path)
            logger.info(f"Loaded segmentation model with {self.model.n_clusters} clusters")
            
            # Load scaler
            scaler_path = self.models_dir / 'scaler.pkl'
            self.scaler = joblib.load(scaler_path)
            
            # Load feature metadata
            metadata_path = self.models_dir / 'feature_metadata.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
            
            # Load PCA transformers
            pca_2d_path = self.models_dir / 'pca_2d.pkl'
            pca_3d_path = self.models_dir / 'pca_3d.pkl'
            
            if pca_2d_path.exists():
                self.pca_2d = joblib.load(pca_2d_path)
            if pca_3d_path.exists():
                self.pca_3d = joblib.load(pca_3d_path)
            
            # Define segment names (can be loaded from config)
            self.segment_names = {
                0: "VIP Customers",
                1: "Occasional Shoppers",
                2: "Regular Loyalists",
                3: "At Risk Customers"
            }
            
            logger.info("All components loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Component not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading components: {str(e)}")
            raise
    
    def prepare_customer_features(self, customer_data):
        """
        Prepare features for a single customer or batch
        
        Args:
            customer_data: DataFrame or dict with customer transaction data
            
        Returns:
            Prepared and scaled features
        """
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        
        # Ensure required columns exist
        required_columns = ['CustomerID', 'InvoiceDate', 'Amount', 'Quantity']
        missing_cols = [col for col in required_columns if col not in customer_data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Process each customer
        prepared_features = []
        customer_ids = []
        
        for customer_id, group in customer_data.groupby('CustomerID'):
            # Calculate RFM features
            recency = self._calculate_recency(group)
            frequency = len(group)
            monetary = group['Amount'].sum()
            
            # Calculate additional features
            avg_transaction = group['Amount'].mean()
            std_transaction = group['Amount'].std()
            total_transactions = len(group)
            total_quantity = group['Quantity'].sum()
            
            # Customer lifetime (simplified)
            first_purchase = group['InvoiceDate'].min()
            last_purchase = group['InvoiceDate'].max()
            customer_lifetime = (last_purchase - first_purchase).days + 1
            
            # Purchase frequency
            unique_days = group['InvoiceDate'].dt.date.nunique()
            purchase_frequency = total_transactions / unique_days if unique_days > 0 else 0
            
            # Create feature vector
            features = {
                'Recency': recency,
                'Frequency': frequency,
                'Monetary': monetary,
                'AvgTransactionValue': avg_transaction,
                'StdTransactionValue': std_transaction,
                'TotalTransactions': total_transactions,
                'TotalQuantity': total_quantity,
                'CustomerLifetime': customer_lifetime,
                'PurchaseFrequency': purchase_frequency
            }
            
            prepared_features.append(features)
            customer_ids.append(customer_id)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(prepared_features, index=customer_ids)
        
        # Apply same transformations as training
        features_df = self._apply_transformations(features_df)
        
        # Select and order features
        available_features = [f for f in self.feature_names if f in features_df.columns]
        features_selected = features_df[available_features]
        
        # Handle missing features
        for feature in self.feature_names:
            if feature not in features_selected.columns:
                logger.warning(f"Feature {feature} not found, filling with 0")
                features_selected[feature] = 0
        
        # Reorder to match training
        features_selected = features_selected[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features_selected)
        features_scaled_df = pd.DataFrame(features_scaled, 
                                          columns=self.feature_names, 
                                          index=customer_ids)
        
        return features_scaled_df
    
    def _calculate_recency(self, customer_transactions):
        """Calculate recency for a customer"""
        # Use today as reference or max from training
        reference_date = pd.Timestamp.now()
        last_purchase = customer_transactions['InvoiceDate'].max()
        recency = (reference_date - last_purchase).days
        return recency
    
    def _apply_transformations(self, features_df):
        """Apply the same transformations as training"""
        # Log transform skewed features
        skewed_features = ['Frequency', 'Monetary', 'AvgTransactionValue', 'TotalTransactions']
        
        for feature in skewed_features:
            if feature in features_df.columns:
                log_feature = f'Log_{feature}'
                if log_feature in self.feature_names:
                    features_df[log_feature] = np.log1p(features_df[feature])
        
        return features_df
    
    def assign_segments(self, customer_data, include_metadata=True):
        """
        Assign segments to new customers
        
        Args:
            customer_data: DataFrame or dict with customer transaction data
            include_metadata: Whether to include additional metadata
            
        Returns:
            DataFrame with segment assignments
        """
        logger.info(f"Assigning segments to {len(customer_data) if hasattr(customer_data, '__len__') else 1} customers")
        
        # Prepare features
        features_scaled = self.prepare_customer_features(customer_data)
        
        # Predict clusters
        clusters = self.model.predict(features_scaled)
        
        # Create results
        results = pd.DataFrame({
            'CustomerID': features_scaled.index,
            'Segment': clusters,
            'Segment_Name': [self.segment_names.get(c, f'Cluster_{c}') for c in clusters]
        })
        
        # Add PCA coordinates if requested
        if include_metadata and self.pca_2d is not None:
            pca_2d_result = self.pca_2d.transform(features_scaled)
            results['PCA1_2D'] = pca_2d_result[:, 0]
            results['PCA2_2D'] = pca_2d_result[:, 1]
            
            if self.pca_3d is not None:
                pca_3d_result = self.pca_3d.transform(features_scaled)
                results['PCA1_3D'] = pca_3d_result[:, 0]
                results['PCA2_3D'] = pca_3d_result[:, 1]
                results['PCA3_3D'] = pca_3d_result[:, 2]
        
        # Add confidence scores (distance to cluster centers)
        distances = self.model.transform(features_scaled)
        results['Distance_to_Center'] = distances.min(axis=1)
        
        # Add feature values for interpretation
        for feature in self.feature_names:
            if feature in features_scaled.columns:
                results[f'Feature_{feature}'] = features_scaled[feature].values
        
        logger.info(f"Segment assignment complete")
        return results
    
    def get_segment_info(self, segment_id=None):
        """
        Get information about segments
        
        Args:
            segment_id: Specific segment ID, or None for all
            
        Returns:
            Dictionary with segment information
        """
        # Load clustered data for statistics
        try:
            clustered_path = Path('../data/features/customer_segments.parquet')
            clustered_data = pd.read_parquet(clustered_path)
            
            segment_info = {}
            
            if segment_id is not None:
                # Get specific segment
                segment_data = clustered_data[clustered_data['Cluster'] == segment_id]
                
                info = {
                    'name': self.segment_names.get(segment_id, f'Cluster_{segment_id}'),
                    'size': len(segment_data),
                    'percentage': len(segment_data) / len(clustered_data) * 100,
                    'avg_recency': segment_data['Recency'].mean(),
                    'avg_frequency': segment_data['Frequency'].mean(),
                    'avg_monetary': segment_data['Monetary'].mean(),
                    'description': self._get_segment_description(segment_id)
                }
                segment_info = info
            else:
                # Get all segments
                for seg_id in sorted(clustered_data['Cluster'].unique()):
                    seg_data = clustered_data[clustered_data['Cluster'] == seg_id]
                    
                    segment_info[seg_id] = {
                        'name': self.segment_names.get(seg_id, f'Cluster_{seg_id}'),
                        'size': len(seg_data),
                        'percentage': len(seg_data) / len(clustered_data) * 100,
                        'avg_recency': seg_data['Recency'].mean(),
                        'avg_frequency': seg_data['Frequency'].mean(),
                        'avg_monetary': seg_data['Monetary'].mean(),
                        'description': self._get_segment_description(seg_id)
                    }
            
            return segment_info
            
        except FileNotFoundError:
            logger.warning("Clustered data not found, returning basic segment info")
            if segment_id is not None:
                return {
                    'name': self.segment_names.get(segment_id, f'Cluster_{segment_id}'),
                    'description': self._get_segment_description(segment_id)
                }
            else:
                return {i: {'name': name, 'description': self._get_segment_description(i)} 
                       for i, name in self.segment_names.items()}
    
    def _get_segment_description(self, segment_id):
        """Get description for a segment"""
        descriptions = {
            0: "High-frequency, high-value customers who purchase regularly and recently",
            1: "Infrequent buyers with moderate recency and lower spend",
            2: "Customers with good purchase frequency and value but not recently active",
            3: "Customers who haven't purchased in a long time (dormant)"
        }
        return descriptions.get(segment_id, "No description available")

# Example usage
if __name__ == "__main__":
    # Initialize assigner
    assigner = SegmentAssigner()
    
    # Example customer data (simulated)
    example_customers = pd.DataFrame({
        'CustomerID': [12345, 67890],
        'InvoiceDate': [pd.Timestamp('2023-12-01'), pd.Timestamp('2023-11-15')],
        'Amount': [500.0, 150.0],
        'Quantity': [5, 3]
    })
    
    # Assign segments
    segments = assigner.assign_segments(example_customers)
    print("\nSegment assignments:")
    print(segments[['CustomerID', 'Segment', 'Segment_Name', 'Distance_to_Center']])
    
    # Get segment information
    print("\nSegment information:")
    segment_info = assigner.get_segment_info()
    for seg_id, info in segment_info.items():
        print(f"\nSegment {seg_id} ({info['name']}):")
        print(f"  Size: {info.get('size', 'N/A')} customers")
        print(f"  Avg Recency: {info.get('avg_recency', 'N/A'):.1f} days")
        print(f"  Avg Frequency: {info.get('avg_frequency', 'N/A'):.1f} purchases")
        print(f"  Description: {info['description']}")