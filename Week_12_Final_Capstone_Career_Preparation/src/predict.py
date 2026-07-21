"""Reusable prediction functions and command-line demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_PATH, RAW_FEATURES


def load_package(model_path: str | Path = MODEL_PATH):
    return joblib.load(model_path)


def predict_customer(payload: dict, model_path: str | Path = MODEL_PATH) -> dict:
    missing = set(RAW_FEATURES) - set(payload)
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    row = pd.DataFrame([{key: payload[key] for key in RAW_FEATURES}])
    package = load_package(model_path)
    probability = float(package["pipeline"].predict_proba(row)[:, 1][0])
    threshold = float(package["threshold"])
    if probability < 0.30:
        level = "Low"
        action = "Standard engagement; no retention discount needed."
    elif probability < 0.60:
        level = "Medium"
        action = "Proactive service check and contract-upgrade message."
    else:
        level = "High"
        action = "Priority retention call with a targeted offer."
    return {
        "churn_probability": round(probability, 4),
        "prediction": int(probability >= threshold),
        "risk_level": level,
        "decision_threshold": round(threshold, 4),
        "recommended_action": action,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict one customer's churn risk")
    parser.add_argument("--input", default="deployment/sample_request.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print(json.dumps(predict_customer(payload), indent=2))


if __name__ == "__main__":
    main()
