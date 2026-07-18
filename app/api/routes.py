from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import pandas as pd
from .schemas import (
    CustomerData,
    CustomerSegmentResponse,
    CustomerProfile,
    SegmentAnalysis,
    RecommendationResponse
)
from ..inference.segmenter import Segmenter
from ..inference.recommendation_engine import RecommendationEngine
from ..inference.customer_profiler import CustomerProfiler
from .dependencies import get_segmenter, get_recommendation_engine, get_profiler

router = APIRouter(prefix="/api/v1", tags=["segmentation"])

@router.post("/segment", response_model=CustomerSegmentResponse)
async def segment_customers(
    customers: List[CustomerData],
    segmenter: Segmenter = Depends(get_segmenter)
):
    """Segment multiple customers"""
    try:
        df = pd.DataFrame([c.dict() for c in customers])
        results = segmenter.segment_customers(df)
        
        return CustomerSegmentResponse(
            status="success",
            total_customers=len(df),
            segment_distribution=results['distribution'],
            segments=results['segments'].tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/customer/{customer_id}", response_model=CustomerProfile)
async def get_customer_profile(
    customer_id: int,
    segmenter: Segmenter = Depends(get_segmenter),
    profiler: CustomerProfiler = Depends(get_profiler)
):
    """Get detailed profile for a specific customer"""
    try:
        profile = profiler.get_profile(customer_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        return CustomerProfile(
            customer_id=customer_id,
            segment=profile['segment'],
            rfm_scores=profile['rfm_scores'],
            lifetime_value=profile['lifetime_value'],
            engagement_score=profile['engagement_score'],
            churn_risk=profile['churn_risk'],
            recommendations=profile.get('recommendations', [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/segment-analysis/{segment_name}", response_model=SegmentAnalysis)
async def analyze_segment(
    segment_name: str,
    segmenter: Segmenter = Depends(get_segmenter)
):
    """Get detailed analysis of a segment"""
    try:
        analysis = segmenter.analyze_segment(segment_name)
        if not analysis:
            raise HTTPException(status_code=404, detail="Segment not found")
        
        return SegmentAnalysis(
            name=analysis['name'],
            size=analysis['size'],
            characteristics=analysis['characteristics'],
            insights=analysis['insights'],
            recommendations=analysis['recommendations']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/{customer_id}", response_model=RecommendationResponse)
async def get_customer_recommendations(
    customer_id: int,
    n_recommendations: int = 10,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get product recommendations for a customer"""
    try:
        recommendations = engine.get_recommendations(customer_id, n_recommendations)
        return RecommendationResponse(
            customer_id=customer_id,
            recommendations=recommendations,
            total=len(recommendations)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-segment")
async def batch_segment(
    file_path: str,
    segmenter: Segmenter = Depends(get_segmenter)
):
    """Segment customers from a file"""
    try:
        df = pd.read_csv(file_path)
        results = segmenter.segment_customers(df)
        
        # Save results
        output_path = f"data/processed/segmented_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_segmented = df.copy()
        df_segmented['Segment'] = results['segments']
        df_segmented.to_csv(output_path, index=False)
        
        return {
            "status": "success",
            "total_customers": len(df),
            "segment_distribution": results['distribution'],
            "output_file": output_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))