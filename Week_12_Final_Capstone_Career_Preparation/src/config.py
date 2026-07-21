"""Shared project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

RANDOM_STATE = 42
TARGET = "Churn"
ID_COLUMN = "CustomerID"
RAW_FEATURES = [
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod",
    "PaperlessBilling",
    "SeniorCitizen",
]
SUCCESS_TARGETS = {
    "Recall": 0.80,
    "ROC_AUC": 0.85,
    "F1": 0.65,
    "Precision": 0.50,
}
