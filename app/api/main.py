from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import uvicorn
import pandas as pd
from datetime import datetime
from .routes import router
from .dependencies import get_segmenter, get_recommendation_engine
from ..inference.segmenter import Segmenter
from ..inference.recommendation_engine import RecommendationEngine
from ..inference.preprocess import Preprocess
from ...src.customer_segmentation.utils.logger import get_logger

# Initialize app
app = FastAPI(
    title="Customer Segmentation API",
    description="API for customer segmentation and analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

# Initialize logger
logger = get_logger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting customer segmentation API")
    # Preload models and data if needed

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down customer segmentation API")

@app.get("/")
async def root():
    return {
        "service": "Customer Segmentation API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/segment/customers")
async def segment_customers(
    customers: List[Dict[str, Any]],
    background_tasks: BackgroundTasks,
    segmenter: Segmenter = Depends(get_segmenter)
):
    """Segment customers based on provided data"""
    try:
        df = pd.DataFrame(customers)
        results = segmenter.segment_customers(df)
        
        # Log segmentation in background
        background_tasks.add_task(
            logger.info,
            f"Segmented {len(df)} customers"
        )
        
        return {
            "status": "success",
            "n_customers": len(df),
            "segments": results['segments'].tolist(),
            "segment_distribution": results['distribution']
        }
    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations")
async def get_recommendations(
    customer_id: int,
    n_recommendations: int = 10,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get recommendations for a customer"""
    try:
        recommendations = engine.get_recommendations(customer_id, n_recommendations)
        return {
            "customer_id": customer_id,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )