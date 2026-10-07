from pathlib import Path
import sys

import pandas as pd

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.customer_segmentation.utils.config import get_config
from src.customer_segmentation.utils.logger import get_logger
from src.customer_segmentation.utils.exceptions import DataValidationError


logger = get_logger(__name__)


def main() -> None:
    config = get_config()
    data_config = config.get_config("data")

    raw_path = PROJECT_ROOT / data_config["raw_path"]
    processed_path = PROJECT_ROOT / data_config["processed_path"]
    required_columns = data_config["required_columns"]

    logger.info("Starting customer data ingestion")
    logger.info(f"Input: {raw_path}")

    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found: {raw_path}")

    # Inspect workbook sheets first
    excel_file = pd.ExcelFile(raw_path)
    logger.info(f"Available sheets: {excel_file.sheet_names}")

    # Online Retail dataset normally uses the first sheet
    df = pd.read_excel(raw_path, sheet_name=excel_file.sheet_names[0])

    logger.info(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")

    # Validate required columns
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {missing_columns}"
        )

    # Basic type normalization
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce"
        )

    if "CustomerID" in df.columns:
        df["CustomerID"] = pd.to_numeric(
            df["CustomerID"],
            errors="coerce"
        )

    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(
            df["Quantity"],
            errors="coerce"
        )

    if "UnitPrice" in df.columns:
        df["UnitPrice"] = pd.to_numeric(
            df["UnitPrice"],
            errors="coerce"
        )

    # Basic quality report
    logger.info(f"Duplicate rows: {df.duplicated().sum():,}")
    logger.info(f"Missing CustomerID: {df['CustomerID'].isna().sum():,}")
    logger.info(f"Missing InvoiceDate: {df['InvoiceDate'].isna().sum():,}")
    logger.info(f"Negative Quantity: {(df['Quantity'] < 0).sum():,}")
    logger.info(f"Non-positive UnitPrice: {(df['UnitPrice'] <= 0).sum():,}")

    # Save ingestion output
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    logger.info(f"Saved ingested data to: {processed_path}")
    logger.info(f"Final ingestion shape: {df.shape[0]:,} x {df.shape[1]}")


if __name__ == "__main__":
    main()
