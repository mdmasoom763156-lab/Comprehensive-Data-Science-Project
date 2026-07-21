"""FastAPI prediction service for the persisted churn model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from src.predict import predict_customer


class CustomerInput(BaseModel):
    Tenure: int = Field(ge=0, le=120)
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)
    Contract: str
    PaymentMethod: str
    PaperlessBilling: str
    SeniorCitizen: int = Field(ge=0, le=1)

    @field_validator("Contract")
    @classmethod
    def valid_contract(cls, value):
        allowed = {"Month-to-month", "One year", "Two year"}
        if value not in allowed:
            raise ValueError(f"Contract must be one of {sorted(allowed)}")
        return value

    @field_validator("PaymentMethod")
    @classmethod
    def valid_payment(cls, value):
        allowed = {"Electronic Check", "Credit Card", "Bank Transfer"}
        if value not in allowed:
            raise ValueError(f"PaymentMethod must be one of {sorted(allowed)}")
        return value

    @field_validator("PaperlessBilling")
    @classmethod
    def valid_paperless(cls, value):
        if value not in {"Yes", "No"}:
            raise ValueError("PaperlessBilling must be Yes or No")
        return value


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Returns churn probability, risk band, and retention action.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "healthy", "model": "churn_pipeline.joblib"}


@app.post("/predict")
def predict(customer: CustomerInput):
    return predict_customer(customer.model_dump())
