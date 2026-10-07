from fastapi import FastAPI

from .routes import router


app = FastAPI(
    title="Customer Segmentation API",
    description="API for customer segmentation and marketing recommendations.",
    version="1.0.0",
)

app.include_router(router)
