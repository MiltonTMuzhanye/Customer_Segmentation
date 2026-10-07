from pathlib import Path

import pandas as pd

from ..utils.logger import get_logger


class SegmentProfiler:
    """Create business profiles for customer segments."""

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

    def __init__(self):
        self.logger = get_logger(__name__)

    def create_profile(
        self,
        customer_features: pd.DataFrame,
        segments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create business-oriented segment profiles."""

        df = customer_features.merge(
            segments,
            on="CustomerID",
            how="inner",
            validate="one_to_one",
        )

        profile = (
            df.groupby("Cluster")
            .agg(
                Customers=("CustomerID", "count"),
                Recency=("Recency", "mean"),
                Frequency=("Frequency", "mean"),
                Monetary=("Monetary", "mean"),
                Customer_Lifetime=("Customer_Lifetime", "mean"),
                Purchase_Frequency=("Purchase_Frequency", "mean"),
                Repeat_Rate=("Repeat_Rate", "mean"),
                Churn_Risk=("Churn_Risk", "mean"),
                Engagement_Score=("Engagement_Score", "mean"),
                Average_Basket_Size=("Average_Basket_Size", "mean"),
            )
            .reset_index()
        )

        profile["Customer_Percentage"] = (
            profile["Customers"] / len(df) * 100
        )

        profile["Segment"] = profile["Cluster"].map(
            self.SEGMENT_NAMES
        )

        profile["Recommended_Action"] = profile["Cluster"].map(
            self.SEGMENT_ACTIONS
        )

        columns = [
            "Cluster",
            "Segment",
            "Customers",
            "Customer_Percentage",
            "Recency",
            "Frequency",
            "Monetary",
            "Customer_Lifetime",
            "Purchase_Frequency",
            "Repeat_Rate",
            "Churn_Risk",
            "Engagement_Score",
            "Average_Basket_Size",
            "Recommended_Action",
        ]

        profile = profile[columns]

        self.logger.info(
            f"Created profiles for {len(profile)} segments"
        )

        return profile

    def save(
        self,
        profile: pd.DataFrame,
        path: str,
    ) -> None:
        """Save segment profiles."""

        output = Path(path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        profile.to_csv(
            output,
            index=False,
        )

        self.logger.info(
            f"Segment profiles saved to: {output}"
        )
