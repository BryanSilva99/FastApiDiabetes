from fastapi import FastAPI, Query
import joblib
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field

from api.database import init_database, list_predictions, save_prediction

app = FastAPI(title="API Diabetes", version="1.0.0")
init_database()


class DiabetesInput(BaseModel):
    Pregnancies: int = Field(ge=0, le=20)
    Glucose: float = Field(gt=0, le=250)
    BloodPressure: float = Field(gt=0, le=150)
    SkinThickness: float = Field(ge=0, le=100)
    Insulin: float = Field(ge=0, le=900)
    BMI: float = Field(gt=0, le=80)
    DiabetesPedigreeFunction: float = Field(ge=0, le=3)
    Age: int = Field(ge=1, le=120)


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

# ruta correcta del modelo
base_path = Path(__file__).resolve().parents[1]
model_path = base_path / "model" / "model.pkl"
metrics_path = base_path / "model" / "metrics.csv"

model = joblib.load(model_path)


@app.get("/")
def home():
    return {"message": "API Diabetes funcionando"}


@app.get("/health")
def health():
    best_model = None

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        best_model = metrics.iloc[0]["model"] if not metrics.empty else None

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "best_model": best_model,
        "database": "predictions.db",
        "features": FEATURE_ORDER,
    }


@app.get("/model/metrics")
def model_metrics():
    if not metrics_path.exists():
        return {"metrics": [], "message": "Aun no existen metricas generadas."}

    metrics = pd.read_csv(metrics_path)
    return {"metrics": metrics.to_dict(orient="records")}


@app.get("/predictions")
def predictions(limit: int = Query(default=20, ge=1, le=100)):
    return {"predictions": list_predictions(limit)}


@app.post("/predict")
def predict(data: DiabetesInput):
    payload = data.model_dump()
    values = pd.DataFrame([[payload[feature] for feature in FEATURE_ORDER]], columns=FEATURE_ORDER)
    prediction = model.predict(values)[0]

    result = {
        "prediction": int(prediction),
        "risk": "alto" if prediction == 1 else "bajo",
        "message": "Alto riesgo de diabetes" if prediction == 1 else "Bajo riesgo",
        "recommendation": (
            "Consulta con un profesional de salud para una evaluacion preventiva."
            if prediction == 1
            else "Manten habitos saludables y controles preventivos regulares."
        ),
    }

    prediction_id = save_prediction(payload, result)

    return {
        "id": prediction_id,
        **result,
    }
