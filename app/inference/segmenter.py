from pathlib import Path

import joblib
import pandas as pd

from src.customer_segmentation.features.engineering import CustomerFeatureEngineer


class CustomerSegmenter:
    """Assign customers to trained customer segments."""

    SEGMENT_NAMES = {
        0: "Dormant / Lost",
        1: "New / One-Time",
        2: "At Risk / Needs Attention",
        3: "Champions / VIP",
        4: "Loyal Customers",
    }

    SEGMENT_ACTIONS = {
        0: "Win-back campaigns and reactivation offers",
        1: "Second-purchase campaigns and onboarding",
        2: "Retention campaigns and targeted incentives",
        3: "VIP treatment, rewards, and premium offers",
        4: "Loyalty rewards, cross-sell, and upsell",
    }

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
    ):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_engineer = CustomerFeatureEngineer()

    def predict(self, customer_features: pd.DataFrame) -> pd.DataFrame:
        """Predict customer segments."""

        customer_ids, clustering_features = (
            self.feature_engineer.create_clustering_matrix(
                customer_features
            )
        )

        scaled_features = self.scaler.transform(
            clustering_features
        )

        scaled_features = pd.DataFrame(
            scaled_features,
            columns=clustering_features.columns,
            index=clustering_features.index,
        )

        clusters = self.model.predict(scaled_features)

        result = customer_ids.copy()
        result["Cluster"] = clusters
        result["Segment"] = result["Cluster"].map(
            self.SEGMENT_NAMES
        )
        result["Recommended_Action"] = result["Cluster"].map(
            self.SEGMENT_ACTIONS
        )

        return result
