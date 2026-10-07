import pandas as pd
import numpy as np

from ..utils.logger import get_logger
from ..utils.exceptions import FeatureEngineeringError


class CustomerFeatureEngineer:
    """Prepare customer-level features for clustering."""

    NUMERIC_FEATURES = [
        "Recency",
        "Frequency",
        "Monetary",
        "Customer_Lifetime",
        "Purchase_Frequency",
        "Repeat_Rate",
        "Churn_Risk",
        "Engagement_Score",
        "Days_Since_First",
        "Average_Basket_Size",
    ]

    LOG_FEATURES = [
        "Frequency",
        "Monetary",
        "Purchase_Frequency",
        "Average_Basket_Size",
    ]

    def __init__(self):
        self.logger = get_logger(__name__)

    def create_clustering_matrix(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create raw and transformed clustering feature matrices."""

        try:
            missing = [
                col
                for col in self.NUMERIC_FEATURES
                if col not in df.columns
            ]

            if missing:
                raise FeatureEngineeringError(
                    f"Missing clustering features: {missing}"
                )

            customer_ids = df[["CustomerID"]].copy()

            features = df[
                self.NUMERIC_FEATURES
            ].copy()

            # Ensure numeric values.
            for col in self.NUMERIC_FEATURES:
                features[col] = pd.to_numeric(
                    features[col],
                    errors="coerce",
                )

            if features.isna().any().any():
                raise FeatureEngineeringError(
                    "Clustering matrix contains missing values."
                )

            # Log transform heavily skewed variables.
            transformed = features.copy()

            for col in self.LOG_FEATURES:
                transformed[col] = np.log1p(
                    transformed[col].clip(lower=0)
                )

            # Guard against infinite values.
            transformed = transformed.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            if transformed.isna().any().any():
                raise FeatureEngineeringError(
                    "Clustering matrix contains invalid values."
                )

            self.logger.info(
                f"Created clustering matrix: "
                f"{transformed.shape[0]:,} customers x "
                f"{transformed.shape[1]} features"
            )

            return customer_ids, transformed

        except FeatureEngineeringError:
            raise

        except Exception as e:
            self.logger.error(
                f"Feature engineering failed: {e}"
            )

            raise FeatureEngineeringError(
                f"Failed to create clustering matrix: {e}"
            )
