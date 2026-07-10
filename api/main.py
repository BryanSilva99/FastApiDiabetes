from __future__ import annotations

import logging

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.database import (
    database_available,
    delete_predictions,
    init_database,
    list_predictions,
    save_prediction,
)
from api.schemas import (
    DeleteHistoryResponse,
    DiabetesInput,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    PredictionsResponse,
    RootResponse,
)
from api.services.model_service import model_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "API academica para estimacion referencial de riesgo de diabetes. "
        "No reemplaza una evaluacion medica profesional."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
init_database()


@app.get("/", response_model=RootResponse, summary="Estado basico de la API")
def home():
    return {"message": "API Diabetes funcionando"}


@app.get("/health", response_model=HealthResponse, summary="Verificar API, modelo y base de datos")
def health():
    db_ok = database_available()
    model_loaded = model_service.loaded

    return {
        "status": "ok" if db_ok and model_loaded else "degraded",
        "api_version": settings.api_version,
        "model_loaded": model_loaded,
        "model_name": model_service.model_name if model_loaded else None,
        "model_version": model_service.model_version if model_loaded else None,
        "database_available": db_ok,
        "feature_order": model_service.feature_order,
    }


@app.get("/model/metrics", response_model=MetricsResponse, summary="Obtener metricas de modelos evaluados")
def model_metrics():
    if not settings.metrics_path.exists():
        return {"metrics": [], "message": "Aun no existen metricas generadas."}

    metrics = pd.read_csv(settings.metrics_path)
    return {"metrics": metrics.to_dict(orient="records"), "message": None}


@app.get("/predictions", response_model=PredictionsResponse, summary="Listar historial de predicciones")
def predictions(
    limit: int = Query(default=settings.history_default_limit, ge=1, le=settings.history_max_limit),
):
    try:
        return {"predictions": list_predictions(limit)}
    except Exception as exc:
        logger.exception("Error consultando historial: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo consultar la base de datos.",
        ) from exc


@app.delete("/predictions", response_model=DeleteHistoryResponse, summary="Eliminar historial completo")
def clear_predictions():
    try:
        deleted = delete_predictions()
        return {"deleted": deleted, "message": "Historial eliminado correctamente."}
    except Exception as exc:
        logger.exception("Error eliminando historial: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo eliminar el historial.",
        ) from exc


@app.post("/predict", response_model=PredictionResponse, summary="Estimar riesgo referencial de diabetes")
def predict(data: DiabetesInput):
    if not model_service.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo de prediccion no esta disponible.",
        )

    payload = data.model_dump()

    try:
        result = model_service.predict(payload)
        saved = save_prediction(payload, result)
    except KeyError as exc:
        logger.exception("Orden de caracteristicas inconsistente: %s", exc)
        raise HTTPException(status_code=500, detail="Error interno en el orden de caracteristicas.") from exc
    except Exception as exc:
        logger.exception("Error ejecutando prediccion: %s", exc)
        raise HTTPException(status_code=503, detail="No se pudo ejecutar la prediccion.") from exc

    return {
        "id": saved["id"],
        "created_at": saved["created_at"],
        **result,
    }
