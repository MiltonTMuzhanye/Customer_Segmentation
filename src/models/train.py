"""
Model training module for customer segmentation system
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
import json

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationModel:
    """Trains and manages customer segmentation models"""
    
    def __init__(self, n_clusters=4, random_state=42):
        """
        Initialize SegmentationModel
        
        Args:
            n_clusters: Number of clusters for KMeans
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.models_dir = Path('../models/')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.kmeans = None
        self.pca_2d = None
        self.pca_3d = None
        self.scaler = None
    
    def load_features(self):
        """
        Load preprocessed features
        
        Returns:
            Tuple of (scaled_features, original_features)
        """
        try:
            # Load scaled features
            features_path = Path('../data/features/customer_features.parquet')
            scaled_features = pd.read_parquet(features_path)
            logger.info(f"Loaded scaled features: {scaled_features.shape}")
            
            # Load original features
            original_path = Path('../data/features/customer_rfm_full.parquet')
            original_features = pd.read_parquet(original_path)
            logger.info(f"Loaded original features: {original_features.shape}")
            
            # Load scaler
            scaler_path = self.models_dir / 'scaler.pkl'
            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler")
            
            return scaled_features, original_features
            
        except FileNotFoundError as e:
            logger.error(f"Feature files not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            raise
    
    def find_optimal_clusters(self, features, max_clusters=10):
        """
        Find optimal number of clusters using multiple metrics
        
        Args:
            features: Scaled feature matrix
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Finding optimal number of clusters (1 to {max_clusters})...")
        
        k_range = range(2, max_clusters + 1)
        results = {
            'k': list(k_range),
            'inertia': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(features)
            
            results['inertia'].append(kmeans.inertia_)
            results['silhouette'].append(silhouette_score(features, kmeans.labels_))
            results['davies_bouldin'].append(davies_bouldin_score(features, kmeans.labels_))
            results['calinski_harabasz'].append(calinski_harabasz_score(features, kmeans.labels_))
            
            logger.info(f"K={k}: Silhouette={results['silhouette'][-1]:.4f}, "
                       f"DB={results['davies_bouldin'][-1]:.4f}")
        
        # Find elbow point (simplified)
        inertias = results['inertia']
        differences = np.diff(inertias)
        differences_pct = differences / inertias[:-1]
        
        # Find where the reduction in inertia slows down
        elbow_k = None
        for i in range(1, len(differences_pct)):
            if differences_pct[i] > 0.7 * differences_pct[i-1]:
                elbow_k = k_range[i]
                break
        
        if elbow_k is None:
            elbow_k = 4  # Default
        
        # Find best silhouette score
        best_silhouette_k = k_range[np.argmax(results['silhouette'])]
        
        logger.info(f"Elbow method suggests K={elbow_k}")
        logger.info(f"Best silhouette score at K={best_silhouette_k}")
        
        return results, elbow_k, best_silhouette_k
    
    def train(self, n_clusters=None):
        """
        Train the segmentation model
        
        Args:
            n_clusters: Number of clusters (uses self.n_clusters if None)
            
        Returns:
            Trained KMeans model
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        logger.info(f"Training KMeans model with {n_clusters} clusters...")
        
        # Load features
        scaled_features, original_features = self.load_features()
        
        # Train KMeans
        self.kmeans = KMeans(n_clusters=n_clusters, 
                            random_state=self.random_state, 
                            n_init=10)
        self.kmeans.fit(scaled_features)
        
        # Evaluate
        silhouette = silhouette_score(scaled_features, self.kmeans.labels_)
        db_index = davies_bouldin_score(scaled_features, self.kmeans.labels_)
        ch_index = calinski_harabasz_score(scaled_features, self.kmeans.labels_)
        
        logger.info(f"Model trained successfully:")
        logger.info(f"  Silhouette Score: {silhouette:.4f}")
        logger.info(f"  Davies-Bouldin Index: {db_index:.4f}")
        logger.info(f"  Calinski-Harabasz Index: {ch_index:.0f}")
        
        # Create PCA for visualization
        self._create_pca_visualizations(scaled_features)
        
        # Save model
        self.save_model()
        
        # Create clustered dataset
        clustered_data = self.create_clustered_dataset(scaled_features, original_features)
        
        return self.kmeans, clustered_data
    
    def _create_pca_visualizations(self, features):
        """Create PCA transformers for visualization"""
        # 2D PCA
        self.pca_2d = PCA(n_components=2, random_state=self.random_state)
        pca_2d_result = self.pca_2d.fit_transform(features)
        logger.info(f"2D PCA explained variance: {self.pca_2d.explained_variance_ratio_.sum():.3f}")
        
        # 3D PCA
        self.pca_3d = PCA(n_components=3, random_state=self.random_state)
        pca_3d_result = self.pca_3d.fit_transform(features)
        logger.info(f"3D PCA explained variance: {self.pca_3d.explained_variance_ratio_.sum():.3f}")
        
        # Save PCA transformers
        joblib.dump(self.pca_2d, self.models_dir / 'pca_2d.pkl')
        joblib.dump(self.pca_3d, self.models_dir / 'pca_3d.pkl')
        logger.info("Saved PCA transformers")
    
    def create_clustered_dataset(self, scaled_features, original_features):
        """
        Create dataset with cluster assignments
        
        Args:
            scaled_features: Scaled feature matrix
            original_features: Original feature DataFrame
            
        Returns:
            DataFrame with cluster assignments
        """
        # Get customers from scaled features (filtered out zero-spend)
        customer_ids = scaled_features.index
        
        # Create clustered DataFrame
        clustered_df = original_features.loc[customer_ids].copy()
        clustered_df['Cluster'] = self.kmeans.labels_
        
        # Save clustered data
        clustered_path = Path('../data/features/customer_segments.parquet')
        clustered_df.to_parquet(clustered_path)
        logger.info(f"Saved clustered data to {clustered_path}")
        
        return clustered_df
    
    def save_model(self):
        """Save the trained model and metadata"""
        # Save KMeans model
        model_path = self.models_dir / 'segmentation_model.pkl'
        joblib.dump(self.kmeans, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save model metadata
        metadata = {
            'model_type': 'KMeans',
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'n_features': self.kmeans.cluster_centers_.shape[1] if self.kmeans else None,
            'inertia': float(self.kmeans.inertia_) if self.kmeans else None,
            'cluster_centers': self.kmeans.cluster_centers_.tolist() if self.kmeans else None
        }
        
        metadata_path = self.models_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model metadata to {metadata_path}")
    
    def compare_algorithms(self, features):
        """
        Compare different clustering algorithms
        
        Args:
            features: Scaled feature matrix
            
        Returns:
            Dictionary with algorithm comparisons
        """
        logger.info("Comparing clustering algorithms...")
        
        algorithms = {
            'KMeans': KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10),
            'Agglomerative': AgglomerativeClustering(n_clusters=self.n_clusters),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
        }
        
        results = {}
        
        for name, model in algorithms.items():
            logger.info(f"Training {name}...")
            
            try:
                if name == 'DBSCAN':
                    labels = model.fit_predict(features)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    if n_clusters > 1:
                        # Filter out noise points for evaluation
                        mask = labels != -1
                        if mask.sum() > 0:
                            silhouette = silhouette_score(features[mask], labels[mask])
                        else:
                            silhouette = None
                    else:
                        silhouette = None
                        
                    results[name] = {
                        'n_clusters': n_clusters,
                        'silhouette': silhouette,
                        'labels': labels
                    }
                    
                else:
                    model.fit(features)
                    labels = model.labels_ if hasattr(model, 'labels_') else model.fit_predict(features)
                    silhouette = silhouette_score(features, labels)
                    
                    results[name] = {
                        'n_clusters': self.n_clusters,
                        'silhouette': silhouette,
                        'labels': labels
                    }
                    
                if silhouette:
                    logger.info(f"  {name}: Silhouette = {silhouette:.4f}")
                    
            except Exception as e:
                logger.error(f"Error with {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize and train model
    model = SegmentationModel(n_clusters=4)
    
    # Load features
    scaled_features, original_features = model.load_features()
    
    # Find optimal clusters
    metrics, elbow_k, best_silhouette_k = model.find_optimal_clusters(scaled_features, max_clusters=10)
    
    # Train with optimal K (using best silhouette)
    trained_model, clustered_data = model.train(n_clusters=best_silhouette_k)
    
    # Compare algorithms
    comparison = model.compare_algorithms(scaled_features)
    
    print(f"\nTraining complete!")
    print(f"Optimal K from silhouette: {best_silhouette_k}")
    print(f"Model silhouette score: {silhouette_score(scaled_features, trained_model.labels_):.4f}")
    
    # Show cluster distribution
    cluster_counts = clustered_data['Cluster'].value_counts().sort_index()
    print(f"\nCluster distribution:")
    for cluster, count in cluster_counts.items():
        percentage = count / len(clustered_data) * 100
        print(f"  Cluster {cluster}: {count} customers ({percentage:.1f}%)")