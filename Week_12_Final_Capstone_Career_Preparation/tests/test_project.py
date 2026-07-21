from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from pypdf import PdfReader

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.config import MODEL_PATH, RAW_FEATURES, SUCCESS_TARGETS
from src.data_validation import load_and_validate
from src.predict import predict_customer


def test_dataset_integrity():
    df = load_and_validate(PROJECT / "data/raw/customer_churn.csv")
    assert df.shape == (500, 9)
    assert df["CustomerID"].is_unique
    assert not df.isna().any().any()
    assert int(df["Churn"].sum()) == 53


def test_split_and_dictionary():
    train = pd.read_csv(PROJECT / "data/processed/train.csv")
    test = pd.read_csv(PROJECT / "data/processed/test.csv")
    dictionary = pd.read_csv(PROJECT / "data/data_dictionary.csv")
    assert len(train) == 375 and len(test) == 125
    assert set(train["CustomerID"]).isdisjoint(set(test["CustomerID"]))
    assert len(dictionary) == 9


def test_model_artifact_and_no_identifier_leakage():
    package = joblib.load(MODEL_PATH)
    assert set(package["raw_features"]) == set(RAW_FEATURES)
    assert "CustomerID" not in package["raw_features"]
    assert "Churn" not in package["raw_features"]
    assert 0.10 <= package["threshold"] <= 0.70


def test_metrics_meet_declared_targets():
    metrics = pd.read_csv(PROJECT / "outputs/model_metrics.csv").iloc[0]
    for key, target in SUCCESS_TARGETS.items():
        assert metrics[key] >= target, (key, metrics[key], target)
    for key in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]:
        assert 0 <= metrics[key] <= 1


def test_prediction_schema_and_repeatability():
    payload = json.loads((PROJECT / "deployment/sample_request.json").read_text())
    first = predict_customer(payload)
    second = predict_customer(payload)
    assert first == second
    assert set(first) == {"churn_probability", "prediction", "risk_level", "decision_threshold", "recommended_action"}
    assert 0 <= first["churn_probability"] <= 1
    assert first["risk_level"] in {"Low", "Medium", "High"}


def test_api_contract():
    from deployment.api import CustomerInput, health
    payload = json.loads((PROJECT / "deployment/sample_request.json").read_text())
    parsed = CustomerInput(**payload)
    assert parsed.Tenure == 8
    assert health()["status"] == "healthy"


def test_notebook_fully_executed():
    notebook = json.loads((PROJECT / "capstone_project.ipynb").read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 10
    assert all(cell["execution_count"] is not None for cell in code_cells)
    errors = [out for cell in code_cells for out in cell.get("outputs", []) if out.get("output_type") == "error"]
    assert not errors


def test_reports_deck_and_visual_evidence():
    for name, minimum_pages in [("technical_documentation.pdf", 6), ("business_report.pdf", 3)]:
        reader = PdfReader(PROJECT / "reports" / name)
        assert len(reader.pages) >= minimum_pages
    assert (PROJECT / "presentation/capstone_presentation.pptx").stat().st_size > 50000
    assert len(list((PROJECT / "screenshots").glob("*.png"))) >= 8


def test_business_impact_is_transparent():
    impact = pd.read_csv(PROJECT / "outputs/business_impact.csv")
    assert len(impact) == 2
    assert impact["Assumption"].str.contains("35% save rate", regex=False).all()
