import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, ClusterMixin
from collections import Counter
from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.exceptions import PredictionError


class EnsembleClusterer(BaseEstimator, ClusterMixin):
    """Ensemble clustering combining multiple clustering algorithms"""
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None, 
                 voting: str = 'soft', random_state: int = 42):
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        self.voting = voting
        self.random_state = random_state
        self.logger = get_logger(__name__)
    
    def fit(self, X: np.ndarray, y=None):
        """Fit all models"""
        self.logger.info(f"Fitting ensemble of {len(self.models)} models")
        
        # Normalize weights
        self.weights = np.array(self.weights) / sum(self.weights)
        
        # Fit all models
        for i, model in enumerate(self.models):
            self.logger.info(f"Fitting model {i+1}/{len(self.models)}")
            model.fit(X)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict clusters using ensemble voting"""
        if not self.models:
            raise PredictionError("No models in ensemble")
        
        self.logger.info(f"Predicting clusters with {len(self.models)} models")
        
        predictions = []
        
        for model in self.models:
            labels = model.predict(X)
            predictions.append(labels)
        
        if self.voting == 'hard':
            # Hard voting: majority vote
            predictions = np.array(predictions).T
            final_labels = np.array([self._majority_vote(row) for row in predictions])
        
        elif self.voting == 'soft':
            # Soft voting: weighted average of cluster assignments
            final_labels = self._soft_voting(predictions)
        
        else:
            raise ValueError(f"Unsupported voting method: {self.voting}")
        
        return final_labels
    
    def _majority_vote(self, row: np.ndarray) -> int:
        """Perform majority voting"""
        # Find the most common label, weighted by model weights
        label_counts = {}
        for i, label in enumerate(row):
            weight = self.weights[i]
            label_counts[label] = label_counts.get(label, 0) + weight
        
        return max(label_counts.items(), key=lambda x: x[1])[0]
    
    def _soft_voting(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Perform soft voting using agreement"""
        n_samples = len(predictions[0])
        n_clusters = max([len(np.unique(p)) for p in predictions])
        final_labels = np.zeros(n_samples, dtype=int)
        
        # Create agreement matrix
        agreement_matrix = np.zeros((n_samples, n_clusters))
        
        for i, pred in enumerate(predictions):
            # Normalize predictions to 0..n_clusters-1
            unique_labels = np.unique(pred)
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            normalized_pred = np.array([label_map[label] for label in pred])
            
            # Add weighted contribution
            for j in range(n_samples):
                agreement_matrix[j, normalized_pred[j]] += self.weights[i]
        
        # Assign samples to clusters with highest agreement
        final_labels = np.argmax(agreement_matrix, axis=1)
        
        return final_labels
    
    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and predict in one step"""
        self.fit(X)
        return self.predict(X)