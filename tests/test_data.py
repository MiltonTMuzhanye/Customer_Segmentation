from pathlib import Path

import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "preprocessed_transactions.csv"


def test_raw_dataset_exists():
    assert RAW_DATA.exists()
    assert RAW_DATA.stat().st_size > 0


def test_preprocessed_dataset_exists():
    assert PROCESSED_DATA.exists()
    assert PROCESSED_DATA.stat().st_size > 0


def test_preprocessed_dataset_schema():
    df = pd.read_csv(PROCESSED_DATA)

    required_columns = {
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
        "Amount",
    }

    assert required_columns.issubset(df.columns)


def test_preprocessed_dataset_has_valid_values():
    df = pd.read_csv(PROCESSED_DATA)

    assert df["CustomerID"].notna().all()
    assert (df["Quantity"] > 0).all()
    assert (df["UnitPrice"] > 0).all()
    assert (df["Amount"] >= 0).all()


def test_preprocessed_dataset_has_customers():
    df = pd.read_csv(PROCESSED_DATA)

    assert df["CustomerID"].nunique() > 0
    assert len(df) > 0
