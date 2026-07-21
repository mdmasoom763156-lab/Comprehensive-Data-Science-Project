"""Reusable, pipeline-safe feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create business-motivated features without learning from the target."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        tenure_safe = frame["Tenure"].clip(lower=1)
        frame["AverageMonthlySpend"] = frame["TotalCharges"] / tenure_safe
        frame["ChargesGap"] = frame["MonthlyCharges"] - frame["AverageMonthlySpend"]
        frame["EstimatedAnnualCharges"] = frame["MonthlyCharges"] * 12
        frame["TenureSquared"] = frame["Tenure"] ** 2
        frame["TenureMonthlyInteraction"] = frame["Tenure"] * frame["MonthlyCharges"]
        frame["IsMonthToMonth"] = (frame["Contract"] == "Month-to-month").astype(int)
        frame["IsAutoPay"] = frame["PaymentMethod"].isin(
            ["Credit Card", "Bank Transfer"]
        ).astype(int)
        frame["IsNewCustomer"] = (frame["Tenure"] <= 12).astype(int)
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        return frame
