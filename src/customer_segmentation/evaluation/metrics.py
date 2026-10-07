import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from ..utils.logger import get_logger


class SegmentationEvaluator:
    """Evaluate customer segmentation quality."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def evaluate(
        self,
        features: pd.DataFrame,
        labels,
    ) -> dict:
        """Calculate clustering quality metrics."""

        if len(features) != len(labels):
            raise ValueError(
                "Features and labels must contain the same number of rows."
            )

        n_clusters = len(set(labels))

        if n_clusters < 2:
            raise ValueError(
                "At least two clusters are required."
            )

        silhouette = silhouette_score(
            features,
            labels,
        )

        calinski = calinski_harabasz_score(
            features,
            labels,
        )

        davies = davies_bouldin_score(
            features,
            labels,
        )

        results = {
            "n_customers": len(features),
            "n_features": features.shape[1],
            "n_clusters": n_clusters,
            "silhouette_score": silhouette,
            "calinski_harabasz_score": calinski,
            "davies_bouldin_score": davies,
        }

        self.logger.info(
            f"Segmentation evaluation complete: "
            f"silhouette={silhouette:.4f}, "
            f"CH={calinski:.2f}, "
            f"DB={davies:.4f}"
        )

        return results

    def save(
        self,
        results: dict,
        path: str,
    ) -> None:
        """Save evaluation metrics."""

        output = pd.DataFrame([results])

        output.to_csv(
            path,
            index=False,
        )

        self.logger.info(
            f"Evaluation metrics saved to: {path}"
        )
