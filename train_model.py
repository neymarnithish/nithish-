from __future__ import annotations

from pathlib import Path
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
BUNDLE_PATH = MODEL_DIR / "niggesh_model_bundle.joblib"
TARGET = "Churn Value"
BUNDLE_VERSION = 2

LEAKAGE_WORDS = (
    "churn score", "churn label", "churn value", "churn reason", "churn status", "churn probability",
    "cancellation reason", "cancel reason",
    "cancellation date", "cancel date", "exit reason", "reason for leaving",
    "after churn", "post churn", "left date", "closed date",
)
ID_WORDS = ("customer id", "member id", "account id", "record id", "subscriber id", "membership id", "user id")
GEO_WORDS = ("country", "state", "city", "zip", "postal", "latitude", "longitude", "lat long")


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        candidates = [DATA_DIR / "Telco_customer_churn.xlsx", DATA_DIR / "Telco_customer_churn.csv"]
        source = next((item for item in candidates if item.exists()), None)
        if source is None:
            raise FileNotFoundError(f"Put a CSV or Excel workbook in {DATA_DIR}")
    else:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Dataset not found: {source}")
    if source.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError("Dataset must be a CSV or Excel workbook.")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def suggest_excluded_columns(frame: pd.DataFrame, target_column: str) -> list[str]:
    suggestions = []
    for column in frame.columns:
        if column == target_column:
            continue
        name = re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip()
        compact = name.replace(" ", "")
        if (
            name in {"id", "count"}
            or any(word in name for word in LEAKAGE_WORDS)
            or any(word in name for word in ID_WORDS)
            or any(word in name for word in GEO_WORDS)
            or compact.endswith("id")
        ):
            suggestions.append(column)
    return suggestions


def guess_positive_label(values) -> str:
    labels = [str(value) for value in values]
    positive_terms = {"1", "1.0", "yes", "true", "churn", "churned", "cancelled", "canceled", "left", "exited", "inactive", "not renewed", "nonrenewed"}
    for label in labels:
        if label.strip().lower() in positive_terms:
            return label
    numeric = pd.to_numeric(pd.Series(labels), errors="coerce")
    if numeric.notna().all():
        return labels[int(numeric.values.argmax())]
    return sorted(labels, key=lambda value: value.lower())[-1]


def _clean_feature_frame(frame: pd.DataFrame, feature_columns: list[str], numeric_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
    X = frame.reindex(columns=feature_columns).copy()
    for column in numeric_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in categorical_columns:
        X[column] = X[column].map(lambda value: np.nan if pd.isna(value) else str(value))
    return X


def _metrics(model, X_test, y_test):
    predicted = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
    }


def train_models_from_frame(
    frame: pd.DataFrame,
    target_column: str,
    positive_label: str | int | float | bool,
    excluded_columns: list[str] | None = None,
) -> dict:
    if target_column not in frame.columns:
        raise ValueError(f"Outcome column '{target_column}' was not found.")
    raw = frame.copy()
    raw.columns = [str(column).strip() for column in raw.columns]
    target_column = str(target_column).strip()
    values = raw[target_column].dropna().unique().tolist()
    if len(values) != 2:
        raise ValueError(f"'{target_column}' must have exactly two outcomes; found {len(values)}.")
    label_text = [str(value) for value in values]
    positive_text = str(positive_label)
    if positive_text not in label_text:
        raise ValueError("The selected churn value is not present in the outcome column.")
    y = (raw[target_column].astype("string") == positive_text).astype("int8")
    valid = raw[target_column].notna()
    raw = raw.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    # Outcome-derived columns reveal the answer and must never be model inputs,
    # even if the user leaves one unchecked in the optional exclusion control.
    forced_leakage = {
        column for column in raw.columns
        if column != target_column
        and any(word in re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip() for word in LEAKAGE_WORDS)
    }
    excluded = sorted((set(excluded_columns or []).intersection(raw.columns) | forced_leakage) - {target_column})
    feature_columns = [column for column in raw.columns if column != target_column and column not in excluded]
    if not feature_columns:
        raise ValueError("No input columns remain after exclusions. Keep at least one customer feature.")

    numeric_columns = []
    categorical_columns = []
    for column in feature_columns:
        series = raw[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric_columns.append(column)
        elif pd.api.types.is_datetime64_any_dtype(series):
            raw[column] = pd.to_datetime(series, errors="coerce").astype("int64").replace(-9223372036854775808, np.nan)
            numeric_columns.append(column)
        else:
            converted = pd.to_numeric(series, errors="coerce")
            non_null = int(series.notna().sum())
            numeric_ratio = int(converted.notna().sum()) / non_null if non_null else 0
            if numeric_ratio >= .9:
                raw[column] = converted
                numeric_columns.append(column)
            else:
                categorical_columns.append(column)
    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable input columns were found.")
    X = _clean_feature_frame(raw, feature_columns, numeric_columns, categorical_columns)
    class_counts = y.value_counts()
    if len(class_counts) != 2 or int(class_counts.min()) < 2:
        raise ValueError("Each outcome needs at least two rows for a train/test comparison.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )
    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    preprocessor = ColumnTransformer(transformers, remainder="drop")

    decision_tree = Pipeline([
        ("preprocess", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=8, min_samples_leaf=15, class_weight="balanced", random_state=42)),
    ])
    # Each model needs a fresh preprocessor instance.
    mlp_preprocessor = ColumnTransformer(transformers, remainder="drop")
    mlp = Pipeline([
        ("preprocess", mlp_preprocessor),
        ("model", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=.001, max_iter=250,
                                 early_stopping=True, n_iter_no_change=15, random_state=42)),
    ])
    decision_tree.fit(X_train, y_train)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mlp.fit(X_train, y_train)

    feature_importance = []
    try:
        names = decision_tree.named_steps["preprocess"].get_feature_names_out()
        values = decision_tree.named_steps["model"].feature_importances_
        feature_importance = sorted(
            [{"feature": str(name).replace("numeric__", "").replace("categorical__", "").replace("_", " "), "importance": float(value)} for name, value in zip(names, values)],
            key=lambda item: item["importance"], reverse=True,
        )
    except Exception:
        pass

    source_name = str(frame.attrs.get("source_name", "uploaded dataset"))
    bundle = {
        "version": BUNDLE_VERSION,
        "decision_tree": decision_tree,
        "mlp": mlp,
        "metrics": {
            "Decision Tree": _metrics(decision_tree, X_test, y_test),
            "Neural Network (MLP)": _metrics(mlp, X_test, y_test),
        },
        "feature_importance": feature_importance,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_column": target_column,
        "positive_label": positive_text,
        "negative_label": next(label for label in label_text if label != positive_text),
        "excluded_columns": excluded,
        "dataset_rows": len(raw),
        "test_size": len(y_test),
        "dataset_name": source_name,
        "mlp_warning": next((str(w.message) for w in caught if "converge" in str(w.message).lower()), None),
    }
    return bundle


def train_models(dataset_path: str | Path | None = None, force: bool = False) -> dict:
    source = Path(dataset_path) if dataset_path else None
    if source is None:
        source = next((path for path in (DATA_DIR / "Telco_customer_churn.xlsx", DATA_DIR / "Telco_customer_churn.csv") if path.exists()), None)
        if source is None:
            raise FileNotFoundError(f"Put a CSV or Excel workbook in {DATA_DIR}")
    if (not force and BUNDLE_PATH.exists() and BUNDLE_PATH.stat().st_mtime >= source.stat().st_mtime):
        try:
            cached = joblib.load(BUNDLE_PATH)
            if cached.get("version") == BUNDLE_VERSION:
                return cached
        except Exception:
            pass
    frame = load_dataset(source)
    frame.attrs["source_name"] = source.name
    target = TARGET if TARGET in frame.columns else next((column for column in frame.columns if "churn" in column.lower()), None)
    if target is None:
        raise ValueError(f"Default Telco dataset must include '{TARGET}'.")
    positive_label = guess_positive_label(frame[target].dropna().unique())
    bundle = train_models_from_frame(frame, target, positive_label, suggest_excluded_columns(frame, target))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH)
    return bundle


def find_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip(): column for column in frame.columns}
    for alias in aliases:
        alias = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
        if alias in normalized:
            return normalized[alias]
    for column in frame.columns:
        name = re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip()
        if any(alias in name for alias in aliases):
            return column
    return None


def score_customers(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    scored = frame.copy()
    feature_columns = bundle["feature_columns"]
    X = _clean_feature_frame(scored, feature_columns, bundle["numeric_columns"], bundle["categorical_columns"])
    probabilities = bundle["decision_tree"].predict_proba(X)[:, 1]
    scored["Churn Probability"] = probabilities
    scored["Risk Score"] = (probabilities * 100).round(1)
    scored["Risk Level"] = np.select([probabilities >= .65, probabilities >= .35], ["High", "Moderate"], default="Low")
    target = bundle["target_column"]
    scored["Observed Churn"] = (scored[target].astype("string") == bundle["positive_label"]).astype("int8")

    tenure_col = find_column(scored, ["tenure months", "membership tenure", "member tenure", "membership length", "months active", "tenure", "lifetime"])
    value_col = find_column(scored, ["cltv", "customer lifetime value", "monthly charges", "monthly fee", "membership fee", "monthly spend", "monthly revenue", "membership price", "monthly price", "revenue", "avg additional charges total", "additional charges total"])
    monthly_names = {"monthly charges", "monthly fee", "membership fee", "monthly spend", "monthly revenue", "membership price", "monthly price"}
    if value_col:
        value = pd.to_numeric(scored[value_col], errors="coerce").fillna(0).clip(lower=0)
        factor = 12 if re.sub(r"[^a-z0-9]+", " ", value_col.lower()).strip() in monthly_names else 1
        scored["Estimated Value at Risk"] = value * scored["Churn Probability"] * factor
    else:
        scored["Estimated Value at Risk"] = np.nan

    if tenure_col and value_col:
        tenure = pd.to_numeric(scored[tenure_col], errors="coerce").fillna(0)
        value = pd.to_numeric(scored[value_col], errors="coerce").fillna(0)
        scored["Segment"] = np.select(
            [scored["Risk Score"] >= 65, (tenure > 48) & (value > value.quantile(.70)), tenure > 48, tenure <= 12],
            ["At-Risk", "Premium", "Loyal", "New"], default="Standard",
        )
    else:
        scored["Segment"] = np.select(
            [scored["Risk Level"] == "High", scored["Risk Level"] == "Moderate"],
            ["At-Risk", "Watchlist"], default="Stable",
        )
    return scored


def main():
    bundle = train_models(force=True)
    print(f"Trained on {bundle['dataset_rows']:,} customer records; evaluated on {bundle['test_size']:,} held-out records.")
    for model, metrics in bundle["metrics"].items():
        print(f"{model}: accuracy={metrics['accuracy']:.1%}, precision={metrics['precision']:.1%}, recall={metrics['recall']:.1%}, F1={metrics['f1']:.1%}")
    print(f"Saved model bundle to {BUNDLE_PATH}")


if __name__ == "__main__":
    main()
