import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, Any, List, Optional
from ..src.customer_segmentation.features.engineering import FeatureEngineer
from ..src.customer_segmentation.utils.logger import get_logger
from ..src.customer_segmentation.utils.config import get_config
from ..src.customer_segmentation.utils.exceptions import PredictionError


class Segmenter:
    """Customer segmentation inference"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.segmentation_config = self.config.get_config('segmentation')
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.segment_profiles = None
        self.encoder = None
        
        # Load artifacts
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load trained models and artifacts"""
        try:
            artifacts_dir = Path('artifacts')
            
            # Load models
            models_dir = artifacts_dir / 'trained_models'
            for model_path in models_dir.glob('*.pkl'):
                model_name = model_path.stem.replace('_model', '')
                model_data = joblib.load(model_path)
                self.models[model_name] = model_data
            
            # Load scaler
            scaler_path = artifacts_dir / 'scalers/standard_scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            # Load feature names
            feature_path = artifacts_dir / 'feature_store/feature_names.pkl'
            if feature_path.exists():
                self.feature_names = joblib.load(feature_path)
            
            # Load segment profiles
            profile_path = artifacts_dir / 'segment_profiles/profiles.pkl'
            if profile_path.exists():
                self.segment_profiles = joblib.load(profile_path)
            
            self.logger.info(f"Loaded {len(self.models)} models and artifacts")
            
        except Exception as e:
            self.logger.error(f"Error loading artifacts: {str(e)}")
            raise
    
    def segment_customers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Segment customers"""
        try:
            self.logger.info(f"Segmenting {len(df)} customers")
            
            # Feature engineering
            feature_engineer = FeatureEngineer()
            feature_df = feature_engineer.create_all_features(df)
            
            # Prepare features
            X = feature_df[self.feature_names].copy()
            X = X.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from best model
            best_model_name = max(self.models.keys())
            model = self.models[best_model_name]
            
            if hasattr(model, 'predict'):
                labels = model.predict(X_scaled)
            else:
                # Fallback to K-Means
                kmeans = self.models.get('kmeans')
                if kmeans:
                    labels = kmeans.predict(X_scaled)
                else:
                    raise PredictionError("No model available for prediction")
            
            # Map cluster labels to segment names
            segment_names = self.map_clusters_to_segments(labels)
            
            # Calculate distribution
            distribution = pd.Series(segment_names).value_counts().to_dict()
            
            self.logger.info(f"Segmentation complete. Distribution: {distribution}")
            
            return {
                'segments': segment_names,
                'distribution': distribution,
                'customer_ids': feature_df['CustomerID'].tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Segmentation error: {str(e)}")
            raise
    
    def map_clusters_to_segments(self, labels: np.ndarray) -> List[str]:
        """Map cluster labels to segment names"""
        # This mapping should be determined during model training
        # For demonstration, we use a predefined mapping
        
        segment_mapping = {
            0: 'Champions',
            1: 'Loyal', 
            2: 'Potential',
            3: 'At Risk',
            4: 'Needs Attention',
            5: 'Dormant'
        }
        
        # If we have stored segment profiles, use them
        if self.segment_profiles:
            # Use stored mapping
            return [self.segment_profiles.get(label, f'Segment_{label}') for label in labels]
        
        # Otherwise, map based on cluster characteristics
        return [segment_mapping.get(label, f'Segment_{label}') for label in labels]
    
    def analyze_segment(self, segment_name: str) -> Optional[Dict[str, Any]]:
        """Get analysis for a specific segment"""
        if self.segment_profiles and segment_name in self.segment_profiles:
            profile = self.segment_profiles[segment_name]
            return {
                'name': segment_name,
                'size': profile.get('size', 0),
                'percentage': profile.get('percentage', 0),
                'characteristics': profile.get('characteristics', {}),
                'insights': self.generate_insights(segment_name, profile),
                'recommendations': self.generate_recommendations(segment_name, profile)
            }
        return None
    
    def generate_insights(self, segment_name: str, profile: Dict[str, Any]) -> List[str]:
        """Generate insights for a segment"""
        insights = []
        chars = profile.get('characteristics', {})
        
        # Recency insight
        if 'recency_mean' in chars:
            recency = chars['recency_mean']
            if recency < 30:
                insights.append(f"Highly recent customers (avg recency: {recency:.0f} days)")
            elif recency < 90:
                insights.append(f"Moderately recent customers (avg recency: {recency:.0f} days)")
            else:
                insights.append(f"Low recency customers (avg recency: {recency:.0f} days)")
        
        # Frequency insight
        if 'frequency_mean' in chars:
            freq = chars['frequency_mean']
            if freq > 10:
                insights.append(f"High purchase frequency (avg: {freq:.1f} orders)")
            elif freq > 5:
                insights.append(f"Medium purchase frequency (avg: {freq:.1f} orders)")
            else:
                insights.append(f"Low purchase frequency (avg: {freq:.1f} orders)")
        
        # Monetary insight
        if 'monetary_mean' in chars:
            monetary = chars['monetary_mean']
            if monetary > 2000:
                insights.append(f"High value customers (avg spend: £{monetary:.2f})")
            elif monetary > 500:
                insights.append(f"Medium value customers (avg spend: £{monetary:.2f})")
            else:
                insights.append(f"Low value customers (avg spend: £{monetary:.2f})")
        
        return insights
    
    def generate_recommendations(self, segment_name: str, profile: Dict[str, Any]) -> List[str]:
        """Generate recommendations for a segment"""
        recommendations = []
        
        # Segment-specific recommendations
        if segment_name == 'Champions':
            recommendations.extend([
                "Offer exclusive VIP benefits and early access to new products",
                "Create a referral program with premium rewards",
                "Personalized product recommendations based on purchase history",
                "Invite to special events and focus groups"
            ])
        elif segment_name == 'Loyal':
            recommendations.extend([
                "Introduce loyalty program with tiered rewards",
                "Cross-sell complementary products",
                "Send personalized thank-you notes and small gifts",
                "Offer early access to sales"
            ])
        elif segment_name == 'Potential':
            recommendations.extend([
                "Provide welcome offers and first-purchase discounts",
                "Educate about product range through targeted content",
                "Offer free samples or trials of new products",
                "Personalized email nurturing campaigns"
            ])
        elif segment_name == 'At Risk':
            recommendations.extend([
                "Send re-engagement campaigns with special offers",
                "Reach out to understand reasons for decreased activity",
                "Offer personalized recommendations to rekindle interest",
                "Consider win-back campaigns with strong incentives"
            ])
        elif segment_name == 'Needs Attention':
            recommendations.extend([
                "Send engagement campaigns to increase activity",
                "Offer educational content about product value",
                "Provide small incentives to encourage repeat purchases",
                "Track engagement and respond to signals of interest"
            ])
        elif segment_name == 'Dormant':
            recommendations.extend([
                "Send re-engagement campaigns to reactivate",
                "Offer significant discounts or special offers",
                "Consider if these customers are worth re-acquiring",
                "Remove from active campaigns to save costs"
            ])
        
        return recommendations