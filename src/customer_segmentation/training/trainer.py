import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import joblib
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from ..models.kmeans import KMeansModel
from ..models.ensemble import EnsembleClusterer
from ..features.engineering import FeatureEngineer
from ..evaluation.metrics import SegmentationMetrics
from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.exceptions import TrainingError


class ModelTrainer:
    """Orchestrate model training pipeline"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.model_config = self.config.get_config('model')
        self.feature_engineer = FeatureEngineer()
        self.metrics = SegmentationMetrics()
        self.models = {}
    
    def prepare_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for training"""
        self.logger.info("Preparing data for training")
        
        # Engineer features
        feature_df = self.feature_engineer.create_all_features(df)
        
        # Separate features for clustering
        cluster_features = [
            'Recency', 'Frequency', 'Monetary',
            'Avg_Order_Value', 'Customer_Lifetime',
            'Purchase_Frequency', 'Engagement_Score',
            'Churn_Risk'
        ]
        
        X = feature_df[cluster_features].copy()
        
        # Handle any missing values
        X = X.fillna(0)
        
        self.logger.info(f"Prepared {len(X)} samples with {len(cluster_features)} features")
        
        return {
            'X': X,
            'feature_names': cluster_features,
            'customer_ids': feature_df['CustomerID'],
            'rfm_data': feature_df
        }
    
    def train_models(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train multiple clustering models"""
        self.logger.info("Training models")
        
        models = {}
        
        # Train K-Means
        self.logger.info("Training K-Means model")
        kmeans = KMeansModel(random_state=self.model_config.get('random_state', 42))
        kmeans.train(X, feature_names)
        models['kmeans'] = kmeans
        
        # Train ensemble
        self.logger.info("Training ensemble model")
        ensemble = EnsembleClusterer(
            models=[kmeans],  # Add more models as needed
            weights=[1.0],
            voting='soft'
        )
        ensemble.fit(X)
        models['ensemble'] = ensemble
        
        self.logger.info(f"Trained {len(models)} models")
        return models
    
    def evaluate_models(self, X: np.ndarray, models: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Evaluate trained models"""
        self.logger.info("Evaluating models")
        
        results = {}
        
        for name, model in models.items():
            self.logger.info(f"Evaluating {name} model")
            labels = model.predict(X)
            
            metrics = self.metrics.calculate_all_metrics(X, labels)
            results[name] = metrics
            
            self.logger.info(f"{name} metrics: {metrics}")
        
        return results
    
    def save_artifacts(self, models: Dict[str, Any], X: np.ndarray, 
                       feature_names: List[str], results: Dict[str, Dict[str, float]]):
        """Save trained models and artifacts"""
        self.logger.info("Saving artifacts")
        
        artifacts_dir = Path('artifacts')
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in models.items():
            model_path = artifacts_dir / f'trained_models/{name}_model.pkl'
            model.save(str(model_path))
        
        # Save scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scaler_path = artifacts_dir / 'scalers/standard_scaler.pkl'
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, str(scaler_path))
        
        # Save feature names
        feature_path = artifacts_dir / 'feature_store/feature_names.pkl'
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(feature_names, str(feature_path))
        
        # Save cluster centers from best model
        best_model_name = max(results, key=lambda x: results[x]['silhouette'])
        best_model = models[best_model_name]
        
        if hasattr(best_model, 'cluster_centers'):
            centers_path = artifacts_dir / 'cluster_centers/centers.pkl'
            centers_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model.cluster_centers, str(centers_path))
        
        self.logger.info("Artifacts saved successfully")
    
    def generate_visualizations(self, X: np.ndarray, labels: np.ndarray, 
                                feature_names: List[str], save_dir: str = 'reports/figures'):
        """Generate visualization for analysis"""
        self.logger.info("Generating visualizations")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # PCA visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Customer Clusters - PCA')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.savefig(save_path / 'clusters_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Cluster characteristics
        self.generate_cluster_profile(X, labels, feature_names, save_path)
        
        # Visualization for each model if multiple
        if len(np.unique(labels)) > 1:
            # Silhouette plot
            self.metrics.plot_silhouette(X, labels, save_path / 'silhouette_plot.png')
    
    def generate_cluster_profile(self, X: np.ndarray, labels: np.ndarray, 
                                 feature_names: List[str], save_path: Path):
        """Generate cluster profile visualization"""
        df = pd.DataFrame(X, columns=feature_names)
        df['Cluster'] = labels
        
        # Calculate cluster means
        cluster_means = df.groupby('Cluster').mean()
        
        # Normalize for heatmap
        cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
        
        plt.figure(figsize=(15, len(cluster_means) * 0.6))
        sns.heatmap(cluster_means_norm, annot=True, fmt='.2f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Normalized Value'})
        plt.title('Cluster Characteristics')
        plt.ylabel('Cluster')
        plt.tight_layout()
        plt.savefig(save_path / 'cluster_profile.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_training_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute full training pipeline"""
        try:
            self.logger.info("Starting training pipeline")
            
            # Prepare data
            prepared_data = self.prepare_data(df)
            X = prepared_data['X'].values
            feature_names = prepared_data['feature_names']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train models
            models = self.train_models(X_scaled, feature_names)
            
            # Evaluate models
            results = self.evaluate_models(X_scaled, models)
            
            # Save artifacts
            self.save_artifacts(models, X_scaled, feature_names, results)
            
            # Generate visualizations
            best_model_name = max(results, key=lambda x: results[x]['silhouette'])
            best_labels = models[best_model_name].predict(X_scaled)
            self.generate_visualizations(X_scaled, best_labels, feature_names)
            
            # Log with MLflow
            with mlflow.start_run():
                mlflow.log_params(self.model_config.get('models', {}))
                for model_name, metrics in results.items():
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{model_name}_{metric_name}", value)
                
                # Log best model
                mlflow.sklearn.log_model(models[best_model_name], "best_model")
            
            self.logger.info("Training pipeline completed successfully")
            
            return {
                'models': models,
                'results': results,
                'best_model': best_model_name,
                'feature_names': feature_names,
                'n_clusters': len(np.unique(best_labels))
            }
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise TrainingError(f"Training pipeline failed: {str(e)}")