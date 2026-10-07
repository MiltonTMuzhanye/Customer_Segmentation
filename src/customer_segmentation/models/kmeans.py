from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans

from ..utils.logger import get_logger


class CustomerKMeans:
    """K-Means customer segmentation model."""

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        n_init: int = 20,
        max_iter: int = 500,
    ):
        self.logger = get_logger(__name__)

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )

    def fit(self, features: pd.DataFrame) -> "CustomerKMeans":
        """Fit K-Means model."""
        self.model.fit(features)

        self.logger.info(
            f"K-Means trained with "
            f"{self.model.n_clusters} clusters"
        )

        return self

    def predict(self, features: pd.DataFrame):
        """Predict customer clusters."""
        return self.model.predict(features)

    def fit_predict(self, features: pd.DataFrame):
        """Fit model and return cluster assignments."""
        return self.model.fit_predict(features)

    def save(self, path: str) -> None:
        """Save trained model."""
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(self.model, path)

        self.logger.info(
            f"K-Means model saved to: {path}"
        )

    def get_cluster_centers(self) -> pd.DataFrame:
        """Return cluster centers."""
        return pd.DataFrame(
            self.model.cluster_centers_
        )
