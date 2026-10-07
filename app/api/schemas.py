from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    CustomerID: int

    Recency: float = Field(ge=0)
    Frequency: float = Field(ge=0)
    Monetary: float = Field(ge=0)

    Customer_Lifetime: float = Field(ge=0)
    Purchase_Frequency: float = Field(ge=0)
    Repeat_Rate: float = Field(ge=0, le=1)
    Churn_Risk: float = Field(ge=0, le=1)
    Engagement_Score: float = Field(ge=0)

    Days_Since_First: float = Field(ge=0)
    Average_Basket_Size: float = Field(ge=0)


class SegmentPrediction(BaseModel):
    CustomerID: int
    Cluster: int
    Segment: str
    Recommended_Action: str


class HealthResponse(BaseModel):
    status: str
    model: str
