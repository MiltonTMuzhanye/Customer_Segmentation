import pandas as pd
import numpy as np
from typing import List
from scipy import stats

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.exceptions import DataProcessingError


class DataPreprocessor:
    """Clean and prepare transaction data for feature engineering."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.data_config = self.config.get_config("data")

    def remove_missing_customer(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)

        df = df[df["CustomerID"].notna()].copy()

        removed = before - len(df)

        self.logger.info(
            f"Removed {removed:,} transactions with missing customer ID"
        )

        return df

    def convert_customer_id(self, df: pd.DataFrame) -> pd.DataFrame:
        if "CustomerID" in df.columns:
            df = df.copy()

            df["CustomerID"] = pd.to_numeric(
                df["CustomerID"],
                errors="coerce",
            )

            df = df[df["CustomerID"].notna()].copy()
            df["CustomerID"] = df["CustomerID"].astype(int)

            self.logger.info(
                f"CustomerID converted to integer. "
                f"Unique customers: {df['CustomerID'].nunique():,}"
            )

        return df

    def remove_cancellations(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)

        df = df[df["Quantity"] > 0].copy()

        removed = before - len(df)

        self.logger.info(
            f"Removed {removed:,} cancelled transactions"
        )

        return df

    def filter_quantity(
        self,
        df: pd.DataFrame,
        min_quantity: int = 1,
    ) -> pd.DataFrame:
        before = len(df)

        df = df[df["Quantity"] >= min_quantity].copy()

        removed = before - len(df)

        self.logger.info(
            f"Removed {removed:,} transactions with "
            f"quantity < {min_quantity}"
        )

        return df

    def remove_outliers_iqr(
        self,
        df: pd.DataFrame,
        columns: List[str],
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        before = len(df)

        for col in columns:
            if col not in df.columns:
                continue

            numeric = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            df = df[
                numeric.between(lower_bound, upper_bound)
            ].copy()

        removed = before - len(df)

        self.logger.info(
            f"Removed {removed:,} outlier rows using IQR"
        )

        return df

    def remove_outliers_zscore(
        self,
        df: pd.DataFrame,
        columns: List[str],
        threshold: float = 3,
    ) -> pd.DataFrame:
        before = len(df)

        for col in columns:
            if col not in df.columns:
                continue

            numeric = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            z_scores = pd.Series(
                np.abs(stats.zscore(numeric, nan_policy="omit")),
                index=df.index,
            )

            df = df[
                z_scores.isna() | (z_scores <= threshold)
            ].copy()

        removed = before - len(df)

        self.logger.info(
            f"Removed {removed:,} outlier rows using Z-score"
        )

        return df

    def create_amount_column(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = df.copy()

        df["Amount"] = (
            df["Quantity"] * df["UnitPrice"]
        )

        self.logger.info(
            f"Created Amount column. "
            f"Range: {df['Amount'].min():.2f} "
            f"to {df['Amount'].max():.2f}"
        )

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete preprocessing pipeline."""

        try:
            self.logger.info(
                "Starting full preprocessing pipeline"
            )

            df = df.copy()

            # Normalize data types.
            df["InvoiceDate"] = pd.to_datetime(
                df["InvoiceDate"],
                errors="coerce",
            )

            df["Quantity"] = pd.to_numeric(
                df["Quantity"],
                errors="coerce",
            )

            df["UnitPrice"] = pd.to_numeric(
                df["UnitPrice"],
                errors="coerce",
            )

            df["CustomerID"] = pd.to_numeric(
                df["CustomerID"],
                errors="coerce",
            )

            # Remove rows with invalid critical values.
            before = len(df)

            df = df.dropna(
                subset=[
                    "InvoiceDate",
                    "Quantity",
                    "UnitPrice",
                    "CustomerID",
                ]
            ).copy()

            self.logger.info(
                f"Removed {before - len(df):,} rows "
                "with invalid critical values"
            )

            preprocess_config = self.data_config.get(
                "preprocessing",
                {},
            )

            if preprocess_config.get(
                "remove_missing_customer",
                True,
            ):
                df = self.remove_missing_customer(df)

            df = self.convert_customer_id(df)

            if preprocess_config.get(
                "remove_cancellations",
                True,
            ):
                df = self.remove_cancellations(df)

            min_quantity = preprocess_config.get(
                "min_quantity",
                1,
            )

            df = self.filter_quantity(
                df,
                min_quantity,
            )

            # Remove non-positive prices.
            before = len(df)

            df = df[df["UnitPrice"] > 0].copy()

            self.logger.info(
                f"Removed {before - len(df):,} transactions "
                "with non-positive UnitPrice"
            )

            df = self.create_amount_column(df)

            outlier_method = preprocess_config.get(
                "outlier_method",
                "none",
            )

            if outlier_method == "iqr":
                df = self.remove_outliers_iqr(
                    df,
                    ["Quantity", "UnitPrice", "Amount"],
                )

            elif outlier_method == "zscore":
                df = self.remove_outliers_zscore(
                    df,
                    ["Quantity", "UnitPrice", "Amount"],
                )

            df = df.reset_index(drop=True)

            self.logger.info(
                f"Preprocessing complete. "
                f"Final shape: {df.shape[0]:,} x {df.shape[1]}"
            )

            return df

        except Exception as e:
            self.logger.error(
                f"Preprocessing failed: {str(e)}"
            )
            raise DataProcessingError(
                f"Failed during preprocessing: {str(e)}"
            )

    def prepare_for_rfm(self, df: pd.DataFrame):
        """Prepare transaction data and calculate RFM reference date."""

        df = df.copy()

        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce",
        )

        reference_date = (
            df["InvoiceDate"].max()
            + pd.Timedelta(days=1)
        )

        self.logger.info(
            f"Reference date for recency: {reference_date}"
        )

        return df, reference_date
