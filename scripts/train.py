from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.customer_segmentation.pipelines.training_pipeline import (
    CustomerSegmentationTrainingPipeline,
)


def main() -> None:
    pipeline = CustomerSegmentationTrainingPipeline(PROJECT_ROOT)
    results = pipeline.run()

    print("\n" + "=" * 60)
    print("CUSTOMER SEGMENTATION TRAINING COMPLETE")
    print("=" * 60)

    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
