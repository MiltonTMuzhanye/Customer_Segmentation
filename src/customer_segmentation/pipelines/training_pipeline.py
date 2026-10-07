from pathlib import Path

import pandas as pd

from ..features.engineering import CustomerFeatureEngineer
from ..features.transformers import CustomerFeatureScaler
from ..models.kmeans import CustomerKMeans
from ..evaluation.profiling import SegmentProfiler
from ..evaluation.metrics import SegmentationEvaluator
from ..utils.logger import get_logger


class CustomerSegmentationTrainingPipeline:
    """End-to-end customer segmentation training pipeline."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.logger = get_logger(__name__)

    def run(self) -> dict:
        """Run segmentation training from processed customer features."""

        self.logger.info("Starting customer segmentation training pipeline")

        processed = self.project_root / "data/processed"
        artifacts = self.project_root / "artifacts"
        reports = self.project_root / "reports"

        customer_features = pd.read_csv(
            processed / "customer_features.csv"
        )

        self.logger.info(
            f"Loaded customer features: "
            f"{customer_features.shape[0]:,} customers"
        )

        engineer = CustomerFeatureEngineer()

        customer_ids, clustering_features = (
            engineer.create_clustering_matrix(
                customer_features
            )
        )

        scaler = CustomerFeatureScaler()

        scaled_features = scaler.fit_transform(
            clustering_features
        )

        scaler_path = (
            artifacts
            / "scalers"
            / "customer_scaler.joblib"
        )

        scaler.save(str(scaler_path))

        model = CustomerKMeans(
            n_clusters=5,
            random_state=42,
            n_init=20,
            max_iter=500,
        )

        labels = model.fit_predict(
            scaled_features
        )

        model_path = (
            artifacts
            / "trained_models"
            / "kmeans_customer_segmentation.joblib"
        )

        model.save(str(model_path))

        segments = customer_ids.copy()
        segments["Cluster"] = labels

        segments_path = (
            processed / "customer_segments.csv"
        )

        segments.to_csv(
            segments_path,
            index=False,
        )

        centers = model.get_cluster_centers()
        centers.columns = clustering_features.columns
        centers.index.name = "Cluster"

        centers_path = (
            artifacts
            / "cluster_centers"
            / "kmeans_centers.csv"
        )

        centers_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        centers.to_csv(
            centers_path
        )

        evaluator = SegmentationEvaluator()

        metrics = evaluator.evaluate(
            scaled_features,
            labels,
        )

        metrics_path = (
            reports
            / "metrics"
            / "final_segmentation_metrics.csv"
        )

        metrics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        evaluator.save(
            metrics,
            str(metrics_path),
        )

        profiler = SegmentProfiler()

        profile = profiler.create_profile(
            customer_features,
            segments,
        )

        profile_path = (
            reports
            / "segment_reports"
            / "segment_profiles.csv"
        )

        profiler.save(
            profile,
            str(profile_path),
        )

        self.logger.info(
            "Customer segmentation training pipeline complete"
        )

        return {
            "customers": len(customer_features),
            "clusters": len(set(labels)),
            "silhouette_score": metrics["silhouette_score"],
            "calinski_harabasz_score": metrics[
                "calinski_harabasz_score"
            ],
            "davies_bouldin_score": metrics[
                "davies_bouldin_score"
            ],
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "segments_path": str(segments_path),
            "profile_path": str(profile_path),
        }