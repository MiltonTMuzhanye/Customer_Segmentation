"""
Pydantic schemas for customer segmentation API
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

class CustomerTransaction(BaseModel):
    """Model for customer transaction data"""
    CustomerID: int = Field(..., description="Unique customer identifier")
    InvoiceDate: str = Field(..., description="Transaction date in ISO format")
    Amount: float = Field(..., gt=0, description="Transaction amount (positive)")
    Quantity: int = Field(..., gt=0, description="Quantity purchased (positive)")
    
    @validator('InvoiceDate')
    def validate_date(cls, v):
        """Validate that the date string can be parsed"""
        try:
            pd.to_datetime(v)
            return v
        except:
            raise ValueError(f"Invalid date format: {v}. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    
    @validator('Amount')
    def validate_amount(cls, v):
        """Validate amount is reasonable"""
        if v > 1000000:  # Arbitrary upper limit
            raise ValueError(f"Amount {v} seems unreasonably high")
        return v
    
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
    transactions: List[CustomerTransaction] = Field(
        ..., 
        description="List of customer transactions"
    )
    include_metadata: bool = Field(
        True, 
        description="Whether to include PCA coordinates and other metadata"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "transactions": [
                    {
                        "CustomerID": 12345,
                        "InvoiceDate": "2023-12-01T10:30:00",
                        "Amount": 150.50,
                        "Quantity": 3
                    }
                ],
                "include_metadata": True
            }
        }

class SegmentInfo(BaseModel):
    """Model for segment information"""
    segment_id: int = Field(..., description="Numeric segment identifier")
    segment_name: str = Field(..., description="Business-friendly segment name")
    description: str = Field(..., description="Description of segment characteristics")
    size: Optional[int] = Field(None, description="Number of customers in segment")
    percentage: Optional[float] = Field(None, ge=0, le=100, description="Percentage of total customers")
    avg_recency: Optional[float] = Field(None, description="Average days since last purchase")
    avg_frequency: Optional[float] = Field(None, description="Average number of purchases")
    avg_monetary: Optional[float] = Field(None, description="Average total spend")
    
    class Config:
        schema_extra = {
            "example": {
                "segment_id": 0,
                "segment_name": "VIP Customers",
                "description": "High-frequency, high-value customers",
                "size": 558,
                "percentage": 12.9,
                "avg_recency": 19.5,
                "avg_frequency": 16.0,
                "avg_monetary": 10007.5
            }
        }

class SegmentAssignment(BaseModel):
    """Model for segment assignment result"""
    CustomerID: int = Field(..., description="Customer identifier")
    Segment: int = Field(..., description="Assigned segment ID")
    Segment_Name: str = Field(..., description="Assigned segment name")
    Distance_to_Center: float = Field(
        ..., 
        description="Distance to cluster center (confidence metric)"
    )
    PCA1_2D: Optional[float] = Field(
        None, 
        description="PCA coordinate for 2D visualization"
    )
    PCA2_2D: Optional[float] = Field(
        None, 
        description="PCA coordinate for 2D visualization"
    )
    PCA1_3D: Optional[float] = Field(
        None, 
        description="PCA coordinate for 3D visualization"
    )
    PCA2_3D: Optional[float] = Field(
        None, 
        description="PCA coordinate for 3D visualization"
    )
    PCA3_3D: Optional[float] = Field(
        None, 
        description="PCA coordinate for 3D visualization"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "CustomerID": 12345,
                "Segment": 0,
                "Segment_Name": "VIP Customers",
                "Distance_to_Center": 1.234,
                "PCA1_2D": 0.567,
                "PCA2_2D": -0.123
            }
        }

class ModelInfo(BaseModel):
    """Model for model information"""
    model_type: str = Field(..., description="Type of model (e.g., KMeans)")
    n_clusters: int = Field(..., description="Number of clusters")
    n_features: Optional[int] = Field(None, description="Number of features used")
    inertia: Optional[float] = Field(None, description="Model inertia score")
    segment_names: Dict[int, str] = Field(..., description="Mapping of segment IDs to names")
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "KMeans",
                "n_clusters": 4,
                "n_features": 6,
                "inertia": 12345.67,
                "segment_names": {
                    0: "VIP Customers",
                    1: "Occasional Shoppers",
                    2: "Regular Loyalists",
                    3: "At Risk Customers"
                }
            }
        }

class BatchSummary(BaseModel):
    """Model for batch processing summary"""
    total_customers: int = Field(..., description="Total customers processed")
    segment_distribution: Dict[str, int] = Field(
        ..., 
        description="Count of customers per segment"
    )
    avg_distance_to_center: float = Field(
        ..., 
        description="Average distance to cluster center"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "total_customers": 100,
                "segment_distribution": {
                    "VIP Customers": 15,
                    "Occasional Shoppers": 35,
                    "Regular Loyalists": 30,
                    "At Risk Customers": 20
                },
                "avg_distance_to_center": 2.345
            }
        }

class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if any")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        """Set timestamp if not provided"""
        return v or datetime.now()
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {"key": "value"},
                "error": None,
                "timestamp": "2023-12-01T10:30:00"
            }
        }

class ErrorResponse(BaseModel):
    """Model for error responses"""
    success: bool = Field(False, description="Always false for error responses")
    message: str = Field(..., description="Error message")
    error: str = Field(..., description="Detailed error description")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "message": "Validation error",
                "error": "Invalid date format",
                "timestamp": "2023-12-01T10:30:00"
            }
        }

# Request/Response models for specific endpoints
class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: datetime = Field(..., description="Check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2023-12-01T10:30:00"
            }
        }

class ExampleDataResponse(BaseModel):
    """Response model for example data"""
    example_transactions: List[CustomerTransaction] = Field(
        ..., 
        description="Example transaction data"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "example_transactions": [
                    {
                        "CustomerID": 12345,
                        "InvoiceDate": "2023-12-01T10:30:00",
                        "Amount": 150.50,
                        "Quantity": 3
                    }
                ]
            }
        }

# Export all schemas
__all__ = [
    'CustomerTransaction',
    'SegmentAssignmentRequest',
    'SegmentInfo',
    'SegmentAssignment',
    'ModelInfo',
    'BatchSummary',
    'APIResponse',
    'ErrorResponse',
    'HealthCheckResponse',
    'ExampleDataResponse'
]