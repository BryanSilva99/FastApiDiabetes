from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "diabetes.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
METRICS_CSV_PATH = BASE_DIR / "model" / "metrics.csv"
METRICS_JSON_PATH = BASE_DIR / "model" / "metrics.json"
RANDOM_STATE = 42


def build_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "SVM": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=1, gamma="scale", random_state=RANDOM_STATE)),
            ]
        ),
        "Regresion Logistica": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Arbol de Decision": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }


def main():
    df = pd.read_csv(DATASET_PATH)

    x = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models()
    results = []

    for name, model in models.items():
        metrics = evaluate_model(name, model, x_train, x_test, y_train, y_test)
        results.append(metrics)

    metrics_df = pd.DataFrame(results).sort_values(
        by=["f1", "recall", "accuracy"],
        ascending=False,
    )

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = models[best_model_name]

    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    metrics_df.to_json(METRICS_JSON_PATH, orient="records", indent=2)
    joblib.dump(best_model, MODEL_PATH)

    print("Comparacion de modelos:")
    print(metrics_df.to_string(index=False))
    print()
    print("Mejor modelo:", best_model_name)
    print("Modelo guardado en:", MODEL_PATH)
    print("Metricas CSV:", METRICS_CSV_PATH)
    print("Metricas JSON:", METRICS_JSON_PATH)


if __name__ == "__main__":
    main()
