from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from api.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


class ModelService:
    def __init__(self, model_path: Path, metadata_path: Path):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        try:
            if self.metadata_path.exists():
                self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            else:
                self.metadata = {}

            self.model = joblib.load(self.model_path)
            if not hasattr(self.model, "predict_proba"):
                logger.error("El modelo cargado no soporta predict_proba.")
                self.model = None
        except Exception as exc:
            logger.exception("No se pudo cargar el modelo: %s", exc)
            self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def feature_order(self) -> list[str]:
        return list(self.metadata.get("feature_order", DEFAULT_FEATURE_ORDER))

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", "unknown"))

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def predict(self, payload: dict[str, int | float]) -> dict[str, int | float | str]:
        if not self.loaded:
            raise RuntimeError("Modelo no disponible")

        values = pd.DataFrame([[payload[feature] for feature in self.feature_order]], columns=self.feature_order)
        probability = float(self.model.predict_proba(values)[0][1])
        prediction = int(probability >= self.threshold)
        risk = classify_risk(probability, self.threshold)

        return {
            "prediction": prediction,
            "risk": risk,
            "probability": probability,
            "risk_percentage": round(probability * 100, 2),
            "threshold": self.threshold,
            "message": build_message(risk),
            "recommendation": build_recommendation(risk),
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


def classify_risk(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "alto"
    if probability >= max(0.35, threshold - 0.15):
        return "medio"
    return "bajo"


def build_message(risk: str) -> str:
    messages = {
        "alto": "Riesgo alto estimado por el modelo",
        "medio": "Riesgo medio estimado por el modelo",
        "bajo": "Riesgo bajo estimado por el modelo",
    }
    return messages[risk]


def build_recommendation(risk: str) -> str:
    recommendations = {
        "alto": "Consulta con un profesional de salud para una evaluacion preventiva.",
        "medio": "Considera mejorar habitos y realizar controles preventivos regulares.",
        "bajo": "Manten habitos saludables y controles preventivos regulares.",
    }
    return recommendations[risk]


settings = get_settings()
model_service = ModelService(settings.model_path, settings.metadata_path)
