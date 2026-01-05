"""
FastAPI application for customer segmentation system
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Optional, Dict, Any
import uvicorn

from src.models.assign_segments import SegmentAssigner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Segmentation API",
    description="API for customer segmentation and profiling",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize segment assigner
try:
    segment_assigner = SegmentAssigner()
    logger.info("Segment assigner initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize segment assigner: {str(e)}")
    segment_assigner = None

# Request/Response models
from pydantic import BaseModel, Field
from typing import List, Optional

class CustomerTransaction(BaseModel):
    """Model for customer transaction data"""
    CustomerID: int
    InvoiceDate: str  # ISO format string
    Amount: float
    Quantity: int
    
    class Config:
        schema_extra = {
            "example": {
                "CustomerID": 12345,
                "InvoiceDate": "2023-12-01T10:30:00",
                "Amount": 150.50,
                "Quantity": 3
            }
        }

class SegmentAssignmentRequest(BaseModel):
    """Request model for segment assignment"""
    transactions: List[CustomerTransaction]
    include_metadata: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "transactions": [
                    {
                        "CustomerID": 12345,
                        "InvoiceDate": "2023-12-01T10:30:00",
                        "Amount": 150.50,
                        "Quantity": 3
                    },
                    {
                        "CustomerID": 12345,
                        "InvoiceDate": "2023-11-15T14:20:00",
                        "Amount": 75.25,
                        "Quantity": 2
                    }
                ],
                "include_metadata": True
            }
        }

class SegmentInfo(BaseModel):
    """Model for segment information"""
    segment_id: int
    segment_name: str
    description: str
    size: Optional[int] = None
    percentage: Optional[float] = None
    avg_recency: Optional[float] = None
    avg_frequency: Optional[float] = None
    avg_monetary: Optional[float] = None

class SegmentAssignment(BaseModel):
    """Model for segment assignment result"""
    CustomerID: int
    Segment: int
    Segment_Name: str
    Distance_to_Center: float
    PCA1_2D: Optional[float] = None
    PCA2_2D: Optional[float] = None

class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Health check endpoint
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint for health check"""
    return {
        "status": "healthy",
        "service": "Customer Segmentation API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if segment_assigner is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "healthy",
        "model_loaded": segment_assigner is not None,
        "timestamp": datetime.now().isoformat()
    }

# Segment assignment endpoint
@app.post("/api/v1/assign-segments", 
          response_model=APIResponse,
          tags=["Segmentation"])
async def assign_segments(request: SegmentAssignmentRequest):
    """
    Assign customer segments based on transaction history
    
    - **transactions**: List of customer transactions
    - **include_metadata**: Whether to include PCA coordinates and other metadata
    """
    try:
        if segment_assigner is None:
            raise HTTPException(status_code=503, detail="Segment assigner not initialized")
        
        logger.info(f"Received request to assign segments for {len(request.transactions)} transactions")
        
        # Convert to DataFrame
        transactions_data = [t.dict() for t in request.transactions]
        df = pd.DataFrame(transactions_data)
        
        # Convert InvoiceDate string to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Assign segments
        assignments = segment_assigner.assign_segments(
            df, 
            include_metadata=request.include_metadata
        )
        
        # Convert to list of dictionaries for response
        assignments_list = assignments.to_dict('records')
        
        # Get segment counts
        segment_counts = assignments['Segment_Name'].value_counts().to_dict()
        
        return APIResponse(
            success=True,
            message=f"Assigned segments for {len(assignments)} customers",
            data={
                "assignments": assignments_list,
                "segment_counts": segment_counts,
                "total_customers": len(assignments)
            }
        )
        
    except ValueError as e:
        logger.error(f"Value error in assign_segments: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in assign_segments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Get segment information endpoint
@app.get("/api/v1/segments", 
         response_model=APIResponse,
         tags=["Segmentation"])
async def get_segments(segment_id: Optional[int] = Query(None, description="Specific segment ID")):
    """
    Get information about customer segments
    
    - **segment_id**: Optional specific segment ID. If not provided, returns all segments.
    """
    try:
        if segment_assigner is None:
            raise HTTPException(status_code=503, detail="Segment assigner not initialized")
        
        # Get segment information
        segment_info = segment_assigner.get_segment_info(segment_id)
        
        # Convert to list for response
        if isinstance(segment_info, dict) and segment_id is None:
            segments_list = []
            for seg_id, info in segment_info.items():
                segments_list.append(SegmentInfo(
                    segment_id=seg_id,
                    segment_name=info.get('name', f'Segment_{seg_id}'),
                    description=info.get('description', ''),
                    size=info.get('size'),
                    percentage=info.get('percentage'),
                    avg_recency=info.get('avg_recency'),
                    avg_frequency=info.get('avg_frequency'),
                    avg_monetary=info.get('avg_monetary')
                ).dict())
        else:
            segments_list = [SegmentInfo(
                segment_id=segment_id,
                segment_name=segment_info.get('name', f'Segment_{segment_id}'),
                description=segment_info.get('description', ''),
                size=segment_info.get('size'),
                percentage=segment_info.get('percentage'),
                avg_recency=segment_info.get('avg_recency'),
                avg_frequency=segment_info.get('avg_frequency'),
                avg_monetary=segment_info.get('avg_monetary')
            ).dict()]
        
        return APIResponse(
            success=True,
            message=f"Retrieved information for {len(segments_list)} segment(s)",
            data={"segments": segments_list}
        )
        
    except Exception as e:
        logger.error(f"Error in get_segments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Model information endpoint
@app.get("/api/v1/model-info", 
         response_model=APIResponse,
         tags=["Model"])
async def get_model_info():
    """Get information about the trained segmentation model"""
    try:
        if segment_assigner is None or segment_assigner.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        model = segment_assigner.model
        
        model_info = {
            "model_type": type(model).__name__,
            "n_clusters": model.n_clusters,
            "n_features": model.cluster_centers_.shape[1] if hasattr(model, 'cluster_centers_') else None,
            "inertia": float(model.inertia_) if hasattr(model, 'inertia_') else None,
            "segment_names": segment_assigner.segment_names
        }
        
        return APIResponse(
            success=True,
            message="Model information retrieved successfully",
            data={"model_info": model_info}
        )
        
    except Exception as e:
        logger.error(f"Error in get_model_info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Batch processing endpoint
@app.post("/api/v1/batch-process", 
          response_model=APIResponse,
          tags=["Batch Processing"])
async def batch_process(file_url: str = Query(..., description="URL to CSV file with transaction data")):
    """
    Process a batch of customers from a CSV file
    
    - **file_url**: URL to CSV file containing transaction data
    """
    try:
        if segment_assigner is None:
            raise HTTPException(status_code=503, detail="Segment assigner not initialized")
        
        # Load data from URL
        logger.info(f"Loading batch data from {file_url}")
        df = pd.read_csv(file_url)
        
        # Validate required columns
        required_columns = ['CustomerID', 'InvoiceDate', 'Amount', 'Quantity']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Assign segments
        assignments = segment_assigner.assign_segments(df, include_metadata=False)
        
        # Calculate summary statistics
        summary = {
            "total_customers": len(assignments),
            "segment_distribution": assignments['Segment_Name'].value_counts().to_dict(),
            "avg_distance_to_center": float(assignments['Distance_to_Center'].mean())
        }
        
        # Convert assignments to CSV for download
        assignments_csv = assignments.to_csv(index=False)
        
        return APIResponse(
            success=True,
            message=f"Processed {len(assignments)} customers from batch",
            data={
                "summary": summary,
                "assignments_csv": assignments_csv,
                "sample_assignments": assignments.head(10).to_dict('records')
            }
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        logger.error(f"Error in batch_process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Example data endpoint
@app.get("/api/v1/example-data", 
         response_model=APIResponse,
         tags=["Examples"])
async def get_example_data():
    """Get example transaction data for testing"""
    example_data = [
        {
            "CustomerID": 12345,
            "InvoiceDate": "2023-12-01T10:30:00",
            "Amount": 150.50,
            "Quantity": 3
        },
        {
            "CustomerID": 12345,
            "InvoiceDate": "2023-11-15T14:20:00",
            "Amount": 75.25,
            "Quantity": 2
        },
        {
            "CustomerID": 67890,
            "InvoiceDate": "2023-10-10T09:15:00",
            "Amount": 300.00,
            "Quantity": 5
        },
        {
            "CustomerID": 67890,
            "InvoiceDate": "2023-09-05T16:45:00",
            "Amount": 120.75,
            "Quantity": 2
        }
    ]
    
    return APIResponse(
        success=True,
        message="Example transaction data",
        data={"example_transactions": example_data}
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=APIResponse(
            success=False,
            message="Error occurred",
            error=exc.detail
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=APIResponse(
            success=False,
            message="Internal server error",
            error="An unexpected error occurred"
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )