import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Dict, Any, Optional, Tuple
import joblib
from pathlib import Path
from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.exceptions import TrainingError


class KMeansModel:
    """K-Means clustering model for customer segmentation"""
    
    def __init__(self, n_clusters: Optional[int] = None, random_state: int = 42):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.model_config = self.config.get_config('model')
        
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.cluster_centers = None
    
    def find_optimal_clusters(self, X: np.ndarray, min_k: int = 2, max_k: int = 15) -> Tuple[int, Dict[int, Dict[str, float]]]:
        """Find optimal number of clusters using multiple metrics"""
        self.logger.info(f"Finding optimal clusters between {min_k} and {max_k}")
        
        results = {}
        best_score = -1
        best_k = min_k
        
        for k in range(min_k, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X)
                
                # Calculate metrics
                silhouette = silhouette_score(X, labels)
                calinski = calinski_harabasz_score(X, labels)
                davies = davies_bouldin_score(X, labels)
                inertia = kmeans.inertia_
                
                results[k] = {
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies,
                    'inertia': inertia,
                    'labels': labels
                }
                
                # Use silhouette score as primary metric
                if silhouette > best_score:
                    best_score = silhouette
                    best_k = k
                
                self.logger.info(f"k={k}: Silhouette={silhouette:.4f}, CH={calinski:.2f}, DB={davies:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error fitting KMeans for k={k}: {str(e)}")
                continue
        
        self.logger.info(f"Optimal clusters: {best_k} (score: {best_score:.4f})")
        return best_k, results
    
    def train(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'KMeansModel':
        """Train the K-Means model"""
        try:
            self.logger.info(f"Training K-Means model with {len(X)} samples")
            self.feature_names = feature_names
            
            # Determine optimal number of clusters if not specified
            if self.n_clusters is None:
                self.n_clusters, results = self.find_optimal_clusters(X)
            
            # Train final model
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            
            self.model.fit(X)
            self.cluster_centers = self.model.cluster_centers_
            
            # Calculate metrics
            labels = self.model.labels_
            metrics = {
                'silhouette': silhouette_score(X, labels),
                'calinski_harabasz': calinski_harabasz_score(X, labels),
                'davies_bouldin': davies_bouldin_score(X, labels),
                'inertia': self.model.inertia_
            }
            
            self.logger.info(f"Training complete. Metrics: {metrics}")
            return self
            
        except Exception as e:
            self.logger.error(f"Error training K-Means model: {str(e)}")
            raise TrainingError(f"K-Means training failed: {str(e)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        if self.model is None:
            raise TrainingError("Model has not been trained yet")
        
        try:
            labels = self.model.predict(X)
            return labels
        except Exception as e:
            self.logger.error(f"Error predicting with K-Means model: {str(e)}")
            raise
    
    def save(self, file_path: str):
        """Save model to disk"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'n_clusters': self.n_clusters,
                'random_state': self.random_state,
                'feature_names': self.feature_names,
                'cluster_centers': self.cluster_centers
            }
            
            joblib.dump(model_data, file_path)
            self.logger.info(f"Model saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, file_path: str):
        """Load model from disk"""
        try:
            model_data = joblib.load(file_path)
            self.model = model_data['model']
            self.n_clusters = model_data['n_clusters']
            self.random_state = model_data['random_state']
            self.feature_names = model_data['feature_names']
            self.cluster_centers = model_data['cluster_centers']
            self.logger.info(f"Model loaded from {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_cluster_characteristics(self, X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """Get cluster characteristics for interpretation"""
        if self.model is None:
            raise TrainingError("Model has not been trained yet")
        
        labels = self.predict(X)
        df = pd.DataFrame(X, columns=feature_names)
        df['Cluster'] = labels
        
        # Calculate cluster characteristics
        characteristics = []
        for cluster in range(self.n_clusters):
            cluster_data = df[df['Cluster'] == cluster]
            mean_values = cluster_data[feature_names].mean()
            std_values = cluster_data[feature_names].std()
            size = len(cluster_data)
            
            characteristics.append({
                'Cluster': cluster,
                'Size': size,
                'Percentage': size / len(df) * 100,
                **{f'{col}_mean': mean_values[col] for col in feature_names},
                **{f'{col}_std': std_values[col] for col in feature_names}
            })
        
        return pd.DataFrame(characteristics)