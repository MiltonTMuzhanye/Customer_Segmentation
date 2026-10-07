import pandas as pd
from typing import Dict, Any, List, Optional

from ..utils.logger import get_logger
from ..utils.exceptions import DataValidationError


class DataValidator:
    """Validate data quality and integrity."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def validate_required_columns(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
    ):
        missing_cols = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_cols:
            raise DataValidationError(
                f"Missing required columns: {missing_cols}"
            )

        self.logger.info(
            f"All required columns present: {required_columns}"
        )

    def validate_data_types(
        self,
        df: pd.DataFrame,
        expected_types: Dict[str, str],
    ):
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                continue

            actual_type = str(df[col].dtype)

            if expected_type not in actual_type:
                self.logger.warning(
                    f"Column '{col}' expected type '{expected_type}' "
                    f"but got '{actual_type}'"
                )

    def validate_range(
        self,
        df: pd.DataFrame,
        column: str,
        min_val: float,
        max_val: float,
    ):
        if column not in df.columns:
            return

        values = pd.to_numeric(df[column], errors="coerce")

        out_of_range = values.notna() & (
            (values < min_val) | (values > max_val)
        )

        count = int(out_of_range.sum())

        if count > 0:
            self.logger.warning(
                f"Column '{column}' has {count} values outside "
                f"range [{min_val}, {max_val}]"
            )

    def validate_missing_values(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
    ):
        missing_percent = df.isnull().sum() / len(df)

        columns_exceeding = missing_percent[
            missing_percent > threshold
        ]

        if len(columns_exceeding) > 0:
            raise DataValidationError(
                "Columns with missing values exceeding "
                f"{threshold}: {dict(columns_exceeding)}"
            )

        self.logger.info(
            f"Missing values validation passed. "
            f"Max missing: {missing_percent.max():.2%}"
        )

    def validate_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
    ):
        duplicate_count = int(df.duplicated(subset=subset).sum())

        if duplicate_count > 0:
            self.logger.warning(
                f"Found {duplicate_count} duplicate rows"
            )

        return duplicate_count

    def validate_customer_id(self, df: pd.DataFrame):
        if "CustomerID" not in df.columns:
            return

        customer_ids = pd.to_numeric(
            df["CustomerID"],
            errors="coerce",
        )

        missing_count = int(customer_ids.isna().sum())

        if missing_count > 0:
            self.logger.warning(
                f"Found {missing_count} null customer IDs. "
                "These rows will be removed during preprocessing."
            )

        negative_ids = customer_ids.notna() & (customer_ids < 0)
        negative_count = int(negative_ids.sum())

        if negative_count > 0:
            self.logger.warning(
                f"Found {negative_count} negative customer IDs"
            )

    def validate_invoice_date(self, df: pd.DataFrame):
        if "InvoiceDate" not in df.columns:
            return

        dates = pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce",
        )

        invalid_dates = int(dates.isna().sum())

        if invalid_dates > 0:
            raise DataValidationError(
                f"Found {invalid_dates} invalid InvoiceDate values"
            )

        min_date = dates.min()
        max_date = dates.max()

        self.logger.info(
            f"Invoice date range: {min_date} to {max_date}"
        )

        future_dates = dates > pd.Timestamp.now()

        if future_dates.any():
            self.logger.warning(
                f"Found {int(future_dates.sum())} future invoice dates"
            )

    def validate_business_rules(self, df: pd.DataFrame):
        if "Quantity" in df.columns:
            quantity = pd.to_numeric(
                df["Quantity"],
                errors="coerce",
            )

            negative_qty = int((quantity < 0).sum())

            if negative_qty > 0:
                self.logger.info(
                    f"Found {negative_qty} negative quantities "
                    "(cancellations)."
                )

        if "UnitPrice" in df.columns:
            prices = pd.to_numeric(
                df["UnitPrice"],
                errors="coerce",
            )

            invalid_prices = int((prices <= 0).sum())

            if invalid_prices > 0:
                self.logger.warning(
                    f"Found {invalid_prices} non-positive prices."
                )

        if all(
            col in df.columns
            for col in ["Quantity", "UnitPrice"]
        ):
            self.logger.info(
                "Quantity and UnitPrice columns available "
                "for amount calculation."
            )

    def validate_all(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
    ):
        try:
            self.logger.info(
                "Starting comprehensive data validation"
            )

            if "required_columns" in config:
                self.validate_required_columns(
                    df,
                    config["required_columns"],
                )

            if "data_types" in config:
                self.validate_data_types(
                    df,
                    config["data_types"],
                )

            if "range_checks" in config:
                for col, ranges in config["range_checks"].items():
                    self.validate_range(
                        df,
                        col,
                        ranges["min"],
                        ranges["max"],
                    )

            self.validate_missing_values(df)
            self.validate_duplicates(df)
            self.validate_customer_id(df)
            self.validate_invoice_date(df)
            self.validate_business_rules(df)

            self.logger.info(
                "All validations completed successfully"
            )

        except DataValidationError as e:
            self.logger.error(
                f"Validation failed: {str(e)}"
            )
            raise

        except Exception as e:
            self.logger.error(
                f"Unexpected error during validation: {str(e)}"
            )
            raise
