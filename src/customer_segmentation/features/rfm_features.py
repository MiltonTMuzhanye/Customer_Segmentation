import pandas as pd
import numpy as np

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.exceptions import FeatureEngineeringError


class RFMFeatureGenerator:
    """Generate Recency, Frequency and Monetary customer features."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.segmentation_config = self.config.get_config(
            "segmentation"
        )

    def calculate_recency(
        self,
        df: pd.DataFrame,
        reference_date: pd.Timestamp,
    ) -> pd.Series:
        last_purchase = (
            df.groupby("CustomerID")["InvoiceDate"]
            .max()
        )

        recency = (
            reference_date - last_purchase
        ).dt.days

        self.logger.info(
            f"Recency range: {recency.min()} to "
            f"{recency.max()} days"
        )

        return recency

    def calculate_frequency(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        frequency = (
            df.groupby("CustomerID")["InvoiceNo"]
            .nunique()
        )

        self.logger.info(
            f"Frequency range: {frequency.min()} to "
            f"{frequency.max()} orders"
        )

        return frequency

    def calculate_monetary(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        monetary = (
            df.groupby("CustomerID")["Amount"]
            .sum()
        )

        self.logger.info(
            f"Monetary range: {monetary.min():.2f} to "
            f"{monetary.max():.2f}"
        )

        return monetary

    @staticmethod
    def _quantile_score(
        series: pd.Series,
        ascending: bool = True,
    ) -> pd.Series:
        """Create robust 1-5 quantile scores."""

        ranked = series.rank(
            method="first",
            ascending=ascending,
        )

        return pd.qcut(
            ranked,
            q=5,
            labels=False,
        ).astype(int) + 1

    def calculate_rfm_scores(
        self,
        df: pd.DataFrame,
        reference_date: pd.Timestamp,
    ) -> pd.DataFrame:

        recency = self.calculate_recency(
            df,
            reference_date,
        )

        frequency = self.calculate_frequency(df)
        monetary = self.calculate_monetary(df)

        rfm = pd.concat(
            [
                recency.rename("Recency"),
                frequency.rename("Frequency"),
                monetary.rename("Monetary"),
            ],
            axis=1,
        ).reset_index()

        # Recency: lower is better -> highest score.
        rfm["R_Score"] = self._quantile_score(
            rfm["Recency"],
            ascending=False,
        )

        # Frequency: higher is better.
        rfm["F_Score"] = self._quantile_score(
            rfm["Frequency"],
            ascending=True,
        )

        # Monetary: higher is better.
        rfm["M_Score"] = self._quantile_score(
            rfm["Monetary"],
            ascending=True,
        )

        rfm["RFM_Score"] = (
            rfm["R_Score"] * 100
            + rfm["F_Score"] * 10
            + rfm["M_Score"]
        )

        rfm["RFM_Segment"] = rfm.apply(
            self.get_rfm_segment,
            axis=1,
        )

        self.logger.info(
            f"RFM scores calculated for "
            f"{len(rfm):,} customers"
        )

        return rfm

    def get_rfm_segment(self, row) -> str:
        r = row["R_Score"]
        f = row["F_Score"]
        m = row["M_Score"]

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"

        if r >= 3 and f >= 3 and m >= 3:
            return "Loyal"

        if r >= 4 and f >= 1 and m >= 1:
            return "Potential"

        if r <= 2 and f >= 4 and m >= 4:
            return "At Risk"

        if r <= 2 and f >= 2 and m >= 2:
            return "Needs Attention"

        return "Dormant"

    def get_customer_tier(self, row) -> str:
        monetary = row["Monetary"]
        frequency = row["Frequency"]
        recency = row["Recency"]

        if (
            monetary > 5000
            and frequency > 10
            and recency < 30
        ):
            return "Platinum"

        if (
            monetary > 2000
            and frequency > 5
            and recency < 60
        ):
            return "Gold"

        if (
            monetary > 500
            and frequency > 3
            and recency < 90
        ):
            return "Silver"

        return "Bronze"

    def calculate_frequency_monetary_ratio(
        self,
        rfm: pd.DataFrame,
    ) -> pd.Series:
        return (
            rfm["Monetary"]
            / rfm["Frequency"].replace(0, np.nan)
        ).fillna(0)

    def create_rfm_features(
        self,
        df: pd.DataFrame,
        reference_date: pd.Timestamp,
    ) -> pd.DataFrame:

        try:
            rfm = self.calculate_rfm_scores(
                df,
                reference_date,
            )

            rfm["Frequency_Monetary_Ratio"] = (
                self.calculate_frequency_monetary_ratio(rfm)
            )

            rfm["Avg_Order_Value"] = (
                rfm["Monetary"]
                / rfm["Frequency"].replace(0, np.nan)
            ).fillna(0)

            rfm["Customer_Tier"] = rfm.apply(
                self.get_customer_tier,
                axis=1,
            )

            rfm["Is_Champion"] = (
                rfm["RFM_Segment"] == "Champions"
            ).astype(int)

            rfm["Is_At_Risk"] = (
                rfm["RFM_Segment"] == "At Risk"
            ).astype(int)

            rfm["Is_Dormant"] = (
                rfm["RFM_Segment"] == "Dormant"
            ).astype(int)

            return rfm

        except Exception as e:
            self.logger.error(
                f"RFM feature generation failed: {e}"
            )
            raise FeatureEngineeringError(
                f"Failed to generate RFM features: {e}"
            )
