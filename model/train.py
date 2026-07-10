from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "diabetes.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_CSV_PATH = MODEL_DIR / "metrics.csv"
METRICS_JSON_PATH = MODEL_DIR / "metrics.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.json"
ROC_CURVE_PATH = MODEL_DIR / "roc_curve.csv"

DATASET_NAME = "Pima Indians Diabetes Dataset"
TARGET_COLUMN = "Outcome"
FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.5
CV_SPLITS = 5


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    expected_columns = set(FEATURE_ORDER + [TARGET_COLUMN])
    missing_columns = expected_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing_columns)}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("La variable objetivo contiene valores faltantes.")

    invalid_targets = set(df[TARGET_COLUMN].unique()).difference({0, 1})
    if invalid_targets:
        raise ValueError(f"La variable objetivo solo debe contener 0 y 1. Valores: {invalid_targets}")

    return df[FEATURE_ORDER + [TARGET_COLUMN]].copy()


def replace_biological_zeros(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    for column in ZERO_AS_MISSING_COLUMNS:
        processed[column] = processed[column].replace(0, float("nan"))
    return processed


def build_preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[("numeric", Pipeline(numeric_steps), FEATURE_ORDER)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "Regresion Logistica": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(scale=True)),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(scale=True)),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1,
                        gamma="scale",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(scale=False)),
                ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(scale=True)),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Arbol de Decision": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(scale=False)),
                ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
            ]
        ),
    }


def evaluate_model(name: str, model: Pipeline, x_train, x_test, y_train, y_test, cv) -> dict[str, float | str]:
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="recall")

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "cv_recall_mean": cv_scores.mean(),
        "cv_recall_std": cv_scores.std(),
    }


def save_evaluation_artifacts(best_model: Pipeline, x_test, y_test) -> dict[str, object]:
    probabilities = best_model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)
    matrix = confusion_matrix(y_test, predictions).tolist()
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, probabilities)

    confusion_payload = {
        "labels": ["no_diabetes", "diabetes"],
        "matrix": matrix,
        "threshold": THRESHOLD,
    }
    CONFUSION_MATRIX_PATH.write_text(json.dumps(confusion_payload, indent=2), encoding="utf-8")

    roc_df = pd.DataFrame(
        {
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "threshold": thresholds,
        }
    )
    roc_df.to_csv(ROC_CURVE_PATH, index=False)

    return confusion_payload


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = replace_biological_zeros(load_dataset())

    x = df[FEATURE_ORDER]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = build_models()
    results = [evaluate_model(name, model, x_train, x_test, y_train, y_test, cv) for name, model in models.items()]

    metrics_df = pd.DataFrame(results).sort_values(
        by=["recall", "f1", "precision", "roc_auc"],
        ascending=False,
    )
    best_model_name = str(metrics_df.iloc[0]["model"])
    best_model = models[best_model_name]
    best_model.fit(x_train, y_train)

    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    metrics_df.to_json(METRICS_JSON_PATH, orient="records", indent=2)
    joblib.dump(best_model, MODEL_PATH)
    confusion_payload = save_evaluation_artifacts(best_model, x_test, y_test)

    best_metrics = metrics_df.iloc[0].to_dict()
    metadata = {
        "model_name": best_model_name,
        "model_version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": DATASET_NAME,
        "target_column": TARGET_COLUMN,
        "feature_order": FEATURE_ORDER,
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "scikit_learn_version": sklearn.__version__,
        "metrics": best_metrics,
        "selection_strategy": (
            "Se prioriza recall de la clase positiva por tratarse de una herramienta preventiva; "
            "se desempata con F1, precision y ROC-AUC para mantener equilibrio entre sensibilidad y falsos positivos."
        ),
        "missing_value_strategy": {
            "zero_as_missing_columns": ZERO_AS_MISSING_COLUMNS,
            "replacement": "Los ceros biologicamente improbables se reemplazan por valores faltantes.",
            "imputation": "Mediana dentro de cada pipeline para evitar fuga de datos.",
            "scaling": "StandardScaler solo en modelos sensibles a escala.",
        },
        "confusion_matrix": confusion_payload,
        "artifacts": {
            "model": str(MODEL_PATH.relative_to(BASE_DIR)),
            "metrics_csv": str(METRICS_CSV_PATH.relative_to(BASE_DIR)),
            "metrics_json": str(METRICS_JSON_PATH.relative_to(BASE_DIR)),
            "confusion_matrix": str(CONFUSION_MATRIX_PATH.relative_to(BASE_DIR)),
            "roc_curve": str(ROC_CURVE_PATH.relative_to(BASE_DIR)),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Comparacion de modelos:")
    print(metrics_df.to_string(index=False))
    print()
    print("Mejor modelo:", best_model_name)
    print("Modelo guardado en:", MODEL_PATH)
    print("Metadata:", METADATA_PATH)


if __name__ == "__main__":
    main()
