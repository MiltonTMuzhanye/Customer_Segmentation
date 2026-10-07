import pandas as pd
import numpy as np

from ..utils.logger import get_logger
from ..utils.exceptions import FeatureEngineeringError


class BehavioralFeatureGenerator:
    """Generate customer behavioral features."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def calculate_customer_lifetime(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        dates = df.groupby("CustomerID")["InvoiceDate"]

        return dates.max() .sub(dates.min()).dt.days

    def calculate_purchase_frequency(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Estimate monthly purchase frequency using unique
        purchase dates, avoiding same-day transaction distortion.
        """

        orders = (
            df[["CustomerID", "InvoiceDate"]]
            .drop_duplicates()
            .assign(
                PurchaseDate=lambda x: x["InvoiceDate"].dt.normalize()
            )
            [["CustomerID", "PurchaseDate"]]
            .drop_duplicates()
            .sort_values(["CustomerID", "PurchaseDate"])
        )

        orders["days_between"] = (
            orders.groupby("CustomerID")["PurchaseDate"]
            .diff()
            .dt.days
        )

        avg_interval = (
            orders.groupby("CustomerID")["days_between"]
            .mean()
        )

        frequency = (
            30 / avg_interval.replace(0, np.nan)
        ).fillna(0)

        return frequency.clip(upper=30)

    def calculate_repeat_rate(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        order_counts = (
            df.groupby("CustomerID")["InvoiceNo"]
            .nunique()
        )

        return (
            order_counts > 1
        ).astype(float)

    def calculate_churn_risk(
        self,
        rfm_df: pd.DataFrame,
        threshold_days: int = 90,
    ) -> pd.Series:
        """
        Normalize recency into a 0-1 churn-risk score.
        """

        return (
            rfm_df["Recency"]
            .div(threshold_days)
            .clip(upper=1.0)
        )

    def calculate_engagement_score(
        self,
        rfm_df: pd.DataFrame,
    ) -> pd.Series:
        recency_max = max(
            float(rfm_df["Recency"].max()),
            1.0,
        )

        frequency_max = max(
            float(rfm_df["Frequency"].max()),
            1.0,
        )

        monetary_max = max(
            float(rfm_df["Monetary"].max()),
            1.0,
        )

        recency_norm = (
            1
            - rfm_df["Recency"] / recency_max
        )

        frequency_norm = (
            rfm_df["Frequency"] / frequency_max
        )

        monetary_norm = (
            rfm_df["Monetary"] / monetary_max
        )

        return (
            recency_norm * 0.3
            + frequency_norm * 0.3
            + monetary_norm * 0.4
        )

    def calculate_average_basket_size(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        quantity = (
            df.groupby("CustomerID")["Quantity"]
            .sum()
        )

        orders = (
            df.groupby("CustomerID")["InvoiceNo"]
            .nunique()
        )

        return (
            quantity
            .div(orders.replace(0, np.nan))
            .fillna(0)
        )

    def create_behavioral_features(
        self,
        df: pd.DataFrame,
        rfm_df: pd.DataFrame,
    ) -> pd.DataFrame:

        try:
            self.logger.info(
                "Creating behavioral features"
            )

            behavioral = rfm_df.copy()
            behavioral = behavioral.set_index(
                "CustomerID"
            )

            behavioral["Customer_Lifetime"] = (
                self.calculate_customer_lifetime(df)
            )

            behavioral["Purchase_Frequency"] = (
                self.calculate_purchase_frequency(df)
            )

            behavioral["Repeat_Rate"] = (
                self.calculate_repeat_rate(df)
            )

            behavioral["Churn_Risk"] = (
                self.calculate_churn_risk(behavioral)
            )

            behavioral["Engagement_Score"] = (
                self.calculate_engagement_score(behavioral)
            )

            first_purchase = (
                df.groupby("CustomerID")["InvoiceDate"]
                .min()
            )

            max_date = df["InvoiceDate"].max()

            behavioral["Days_Since_First"] = (
                max_date - first_purchase
            ).dt.days

            behavioral["Average_Basket_Size"] = (
                self.calculate_average_basket_size(df)
            )

            behavioral = behavioral.reset_index()

            self.logger.info(
                f"Created behavioral features for "
                f"{len(behavioral):,} customers"
            )

            return behavioral

        except Exception as e:
            self.logger.error(
                f"Behavioral feature generation failed: {e}"
            )

            raise FeatureEngineeringError(
                f"Failed to generate behavioral features: {e}"
            )
