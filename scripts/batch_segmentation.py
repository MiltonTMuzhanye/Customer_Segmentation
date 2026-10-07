from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.inference.segmenter import CustomerSegmenter


def main() -> None:
    processed = PROJECT_ROOT / "data/processed"
    artifacts = PROJECT_ROOT / "artifacts"

    customer_features = pd.read_csv(
        processed / "customer_features.csv"
    )

    segmenter = CustomerSegmenter(
        model_path=str(
            artifacts
            / "trained_models"
            / "kmeans_customer_segmentation.joblib"
        ),
        scaler_path=str(
            artifacts
            / "scalers"
            / "customer_scaler.joblib"
        ),
    )

    results = segmenter.predict(customer_features)

    output_path = (
        processed / "batch_segment_predictions.csv"
    )

    results.to_csv(output_path, index=False)

    print("\nBatch segmentation complete")
    print(f"Customers: {len(results):,}")
    print(f"Output: {output_path}")

    print("\nSegment distribution:")
    print(results["Segment"].value_counts())

    print("\nSample:")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
