from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from .schemas import (
    CustomerFeatures,
    SegmentPrediction,
    HealthResponse,
)
from app.inference.segmenter import CustomerSegmenter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

segmenter = CustomerSegmenter(
    model_path=str(
        PROJECT_ROOT
        / "artifacts"
        / "trained_models"
        / "kmeans_customer_segmentation.joblib"
    ),
    scaler_path=str(
        PROJECT_ROOT
        / "artifacts"
        / "scalers"
        / "customer_scaler.joblib"
    ),
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "healthy",
        "model": "kmeans_customer_segmentation",
    }


@router.get("/segments")
def segments():
    return {
        "segments": segmenter.SEGMENT_NAMES,
        "actions": segmenter.SEGMENT_ACTIONS,
    }


@router.post(
    "/predict",
    response_model=SegmentPrediction,
)
def predict(customer: CustomerFeatures):
    try:
        features = pd.DataFrame([customer.model_dump()])

        result = segmenter.predict(features)

        return result.iloc[0].to_dict()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation prediction failed: {exc}",
        )
