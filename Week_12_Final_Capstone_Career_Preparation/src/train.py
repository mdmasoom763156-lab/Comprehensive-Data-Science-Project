"""Train, tune, evaluate, explain, and persist the churn model."""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-week12")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    ID_COLUMN,
    METADATA_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    RAW_FEATURES,
    SCREENSHOT_DIR,
    SUCCESS_TARGETS,
    TARGET,
)
from .data_validation import file_sha256, load_and_validate, quality_summary
from .features import ChurnFeatureEngineer


ENGINEERED_NUMERIC = [
    "Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "AverageMonthlySpend", "ChargesGap", "EstimatedAnnualCharges",
    "TenureSquared", "TenureMonthlyInteraction", "IsMonthToMonth",
    "IsAutoPay", "IsNewCustomer",
]
CATEGORICAL = ["Contract", "PaymentMethod", "PaperlessBilling"]


def build_preprocessor():
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, ENGINEERED_NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(model):
    return Pipeline(
        [
            ("features", ChurnFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def metric_row(name, y_true, probability, threshold=0.5):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
    return {
        "Model": name,
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(y_true, prediction),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probability),
        "PR_AUC": average_precision_score(y_true, probability),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def choose_threshold(y_true, probabilities):
    rows = []
    for threshold in np.arange(0.10, 0.71, 0.02):
        pred = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        score = (0.60 * f1) + (0.40 * recall)
        rows.append(
            {
                "Threshold": float(threshold),
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Business_Score": score,
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[(table["Recall"] >= 0.80) & (table["Precision"] >= 0.50)]
    selection = eligible if not eligible.empty else table
    best = selection.sort_values(["Business_Score", "Recall", "Precision"], ascending=False).iloc[0]
    return float(best["Threshold"]), table


def make_plots(df, comparison, metrics, y_test, probability, feature_importance, impact):
    sns.set_theme(style="whitegrid", context="notebook")
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    counts = df[TARGET].map({0: "Stay", 1: "Churn"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0], palette=["#2f6bff", "#ff6b4a"], hue=counts.index, legend=False)
    axes[0].set_title("Customer outcome distribution")
    axes[0].set_ylabel("Customers")
    axes[0].set_xlabel("")
    for i, value in enumerate(counts.values):
        axes[0].text(i, value + 5, str(value), ha="center", fontweight="bold")
    quality = quality_summary(df)
    axes[1].axis("off")
    axes[1].text(0.02, 0.92, "Validated dataset", fontsize=18, fontweight="bold")
    facts = [
        f"Rows: {len(df):,}",
        f"Columns: {df.shape[1]}",
        f"Missing values: {int(df.isna().sum().sum())}",
        f"Duplicate IDs: {int(df[ID_COLUMN].duplicated().sum())}",
        f"Churn rate: {df[TARGET].mean():.1%}",
    ]
    axes[1].text(0.02, 0.75, "\n".join(facts), fontsize=15, linespacing=1.8, va="top")
    fig.suptitle("Data quality and target balance", fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "01_data_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.boxplot(data=df, x=TARGET, y="Tenure", ax=axes[0], palette=["#2f6bff", "#ff6b4a"], hue=TARGET, legend=False)
    axes[0].set_xticks([0, 1], ["Stay", "Churn"])
    axes[0].set_title("Churners have shorter tenure")
    contract = df.groupby("Contract")[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=contract.values, y=contract.index, ax=axes[1], color="#2f6bff")
    axes[1].set_title("Churn rate by contract")
    axes[1].set_xlabel("Churn rate")
    axes[1].xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    fig.suptitle("Exploratory analysis identifies retention signals", fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "02_eda_drivers.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    chart = comparison.melt(id_vars="Model", value_vars=["ROC_AUC", "PR_AUC", "F1", "Recall"], var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=chart, x="Score", y="Model", hue="Metric", ax=ax, palette="Blues")
    ax.set_xlim(0, 1.05)
    ax.set_title("Model comparison on the untouched test set", fontsize=18, fontweight="bold")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "03_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    prediction = (probability >= metrics["Threshold"]).astype(int)
    cm = confusion_matrix(y_test, prediction)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xticklabels(["Stay", "Churn"])
    ax.set_yticklabels(["Stay", "Churn"], rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Holdout confusion matrix at threshold {metrics['Threshold']:.2f}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "04_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probability)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, probability)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(fpr, tpr, color="#2f6bff", linewidth=3, label=f"AUC = {metrics['ROC_AUC']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].legend()
    axes[1].plot(pr_recall, pr_precision, color="#ff6b4a", linewidth=3, label=f"AP = {metrics['PR_AUC']:.3f}")
    axes[1].axhline(df[TARGET].mean(), linestyle="--", color="gray", label="Base rate")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1].legend()
    fig.suptitle("The selected model separates churn risk strongly", fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "05_roc_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    top = feature_importance.head(12).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9.5, 6))
    sns.barplot(data=top, x="Importance", y="Feature", ax=ax, color="#2f6bff")
    ax.set_title("Top churn prediction drivers", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "06_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    impact_plot = impact.set_index("Scenario")["Estimated_Net_Value_INR"]
    colors = ["#9aa6b2", "#2f6bff"]
    bars = ax.bar(impact_plot.index, impact_plot.values, color=colors)
    ax.set_ylabel("Estimated net value (INR)")
    ax.set_title("Illustrative retention campaign economics", fontsize=18, fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, f"INR {bar.get_height():,.0f}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "07_business_impact.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_project(data_path=RAW_DATA_PATH):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_and_validate(data_path)
    X = df[RAW_FEATURES].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    train_export = df.loc[X_train.index].sort_index()
    test_export = df.loc[X_test.index].sort_index()
    processed = RAW_DATA_PATH.parent.parent / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    train_export.to_csv(processed / "train.csv", index=False)
    test_export.to_csv(processed / "test.csv", index=False)
    quality_summary(df).to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "Gradient Boosting": make_pipeline(
            GradientBoostingClassifier(random_state=RANDOM_STATE)
        ),
    }
    candidate_rows = []
    candidate_probabilities = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        probability = pipeline.predict_proba(X_test)[:, 1]
        candidate_probabilities[name] = probability
        candidate_rows.append(metric_row(name, y_test, probability, 0.5))

    rf = make_pipeline(
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
    )
    grid = {
        "model__n_estimators": [150, 300],
        "model__max_depth": [None, 6, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt"],
    }
    search = GridSearchCV(
        rf,
        grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    oof_probability = cross_val_predict(
        best_model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    threshold, threshold_table = choose_threshold(y_train, oof_probability)
    test_probability = best_model.predict_proba(X_test)[:, 1]
    tuned_metrics = metric_row("Tuned Random Forest", y_test, test_probability, threshold)
    candidate_rows.append(tuned_metrics)
    comparison = pd.DataFrame(candidate_rows)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    pd.DataFrame([tuned_metrics]).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    threshold_table.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(20).to_csv(
        OUTPUT_DIR / "hyperparameter_results.csv", index=False
    )

    prediction = (test_probability >= threshold).astype(int)
    predictions = test_export[[ID_COLUMN, TARGET]].copy()
    predictions["ChurnProbability"] = test_probability
    predictions["PredictedChurn"] = prediction
    predictions["RiskLevel"] = pd.cut(
        predictions["ChurnProbability"],
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low", "Medium", "High"],
    )
    predictions.sort_values("ChurnProbability", ascending=False).to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False
    )

    preprocessor = best_model.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame(
        {
            "Feature": names,
            "Importance": best_model.named_steps["model"].feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    scaled_factor = len(df) / len(y_test)
    tp_scaled = tuned_metrics["TP"] * scaled_factor
    fp_scaled = tuned_metrics["FP"] * scaled_factor
    actual_churn_scaled = int(y_test.sum()) * scaled_factor
    value_per_saved_customer = 12000
    save_rate = 0.35
    offer_cost = 1200
    targeted_value = tp_scaled * save_rate * value_per_saved_customer - (tp_scaled + fp_scaled) * offer_cost
    blanket_value = actual_churn_scaled * save_rate * value_per_saved_customer - len(df) * offer_cost
    impact = pd.DataFrame(
        [
            {
                "Scenario": "Blanket campaign",
                "Customers_Contacted": int(len(df)),
                "Estimated_Churners_Reached": int(round(actual_churn_scaled)),
                "Estimated_Net_Value_INR": float(blanket_value),
            },
            {
                "Scenario": "Model-targeted campaign",
                "Customers_Contacted": int(round((tuned_metrics["TP"] + tuned_metrics["FP"]) * scaled_factor)),
                "Estimated_Churners_Reached": int(round(tp_scaled)),
                "Estimated_Net_Value_INR": float(targeted_value),
            },
        ]
    )
    impact["Assumption"] = "35% save rate; INR 12,000 retained value; INR 1,200 offer cost"
    impact.to_csv(OUTPUT_DIR / "business_impact.csv", index=False)

    package = {
        "pipeline": best_model,
        "threshold": threshold,
        "raw_features": RAW_FEATURES,
        "risk_boundaries": {"low_max": 0.30, "medium_max": 0.60},
    }
    joblib.dump(package, MODEL_PATH)
    metadata = {
        "project": "Customer Churn Prediction and Retention Strategy",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "dataset_columns": int(df.shape[1]),
        "dataset_sha256": file_sha256(data_path),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_rate": float(y.mean()),
        "selected_model": "Tuned Random Forest",
        "decision_threshold": threshold,
        "best_params": search.best_params_,
        "cross_validation_best_roc_auc": float(search.best_score_),
        "test_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v for k, v in tuned_metrics.items() if k != "Model"},
        "success_targets": SUCCESS_TARGETS,
        "success_targets_met": {
            key: bool(tuned_metrics[key] >= target) for key, target in SUCCESS_TARGETS.items()
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "Small 500-row educational dataset.",
            "No live production labels or causal retention experiment.",
            "Financial impact is an illustrative scenario, not realized revenue.",
            "Monitor feature drift, subgroup performance, and calibration before production use.",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_plots(df, comparison, tuned_metrics, y_test, test_probability, importance, impact)
    return {
        "data": df,
        "comparison": comparison,
        "metrics": tuned_metrics,
        "threshold_table": threshold_table,
        "feature_importance": importance,
        "impact": impact,
        "predictions": predictions,
        "metadata": metadata,
    }


if __name__ == "__main__":
    result = train_project()
    print(json.dumps(result["metadata"], indent=2))
