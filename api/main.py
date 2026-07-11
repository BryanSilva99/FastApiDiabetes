from __future__ import annotations

import logging

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.database import (
    database_available,
    delete_prediction,
    delete_predictions,
    delete_profile,
    get_prediction,
    get_profile,
    init_database,
    list_predictions,
    save_prediction,
    save_profile,
    update_prediction,
    update_profile,
)
from api.schemas import (
    DeleteResponse,
    DeleteHistoryResponse,
    DiabetesInput,
    HealthResponse,
    MetricsResponse,
    PredictionHistoryItem,
    PredictionResponse,
    PredictionsResponse,
    ProfileCreateRequest,
    ProfileResponse,
    ProfileUpdateRequest,
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


@app.get("/predictions/{prediction_id}", response_model=PredictionHistoryItem, summary="Consultar detalle de una prediccion")
def prediction_detail(prediction_id: int):
    try:
        prediction = get_prediction(prediction_id)
    except Exception as exc:
        logger.exception("Error consultando prediccion %s: %s", prediction_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo consultar la base de datos.",
        ) from exc

    if prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediccion no encontrada.")

    return prediction


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


def _run_prediction(data: DiabetesInput):
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
        "updated_at": saved["updated_at"],
        **result,
    }


@app.post("/predict", response_model=PredictionResponse, summary="Estimar riesgo referencial de diabetes")
def predict(data: DiabetesInput):
    return _run_prediction(data)


@app.post("/predictions", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED, summary="Crear una evaluacion")
def create_prediction(data: DiabetesInput):
    return _run_prediction(data)


@app.put("/predictions/{prediction_id}", response_model=PredictionHistoryItem, summary="Editar y recalcular una prediccion")
def edit_prediction(prediction_id: int, data: DiabetesInput):
    if not model_service.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo de prediccion no esta disponible.",
        )

    payload = data.model_dump()

    try:
        if get_prediction(prediction_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediccion no encontrada.")

        result = model_service.predict(payload)
        updated = update_prediction(prediction_id, payload, result)
    except HTTPException:
        raise
    except KeyError as exc:
        logger.exception("Orden de caracteristicas inconsistente: %s", exc)
        raise HTTPException(status_code=500, detail="Error interno en el orden de caracteristicas.") from exc
    except Exception as exc:
        logger.exception("Error actualizando prediccion: %s", exc)
        raise HTTPException(status_code=503, detail="No se pudo actualizar la prediccion.") from exc

    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediccion no encontrada.")

    return updated


@app.delete("/predictions/{prediction_id}", response_model=DeleteResponse, summary="Eliminar una prediccion")
def remove_prediction(prediction_id: int):
    try:
        deleted = delete_prediction(prediction_id)
    except Exception as exc:
        logger.exception("Error eliminando prediccion %s: %s", prediction_id, exc)
        raise HTTPException(status_code=503, detail="No se pudo eliminar la prediccion.") from exc

    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediccion no encontrada.")

    return {"deleted": True, "message": "Prediccion eliminada correctamente."}


@app.post("/profile", response_model=ProfileResponse, status_code=status.HTTP_201_CREATED, summary="Crear perfil academico")
def create_profile(data: ProfileCreateRequest):
    try:
        if get_profile() is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Ya existe un perfil.")
        return save_profile(data.model_dump())
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error creando perfil: %s", exc)
        raise HTTPException(status_code=503, detail="No se pudo crear el perfil.") from exc


@app.get("/profile", response_model=ProfileResponse, summary="Consultar perfil academico")
def read_profile():
    try:
        profile = get_profile()
    except Exception as exc:
        logger.exception("Error consultando perfil: %s", exc)
        raise HTTPException(status_code=503, detail="No se pudo consultar el perfil.") from exc

    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Perfil no encontrado.")

    return profile


@app.put("/profile", response_model=ProfileResponse, summary="Actualizar perfil academico")
def edit_profile(data: ProfileUpdateRequest):
    try:
        profile = update_profile(data.model_dump(exclude_unset=True))
    except Exception as exc:
        logger.exception("Error actualizando perfil: %s", exc)
        raise HTTPException(status_code=503, detail="No se pudo actualizar el perfil.") from exc

    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Perfil no encontrado.")

    return profile


@app.delete("/profile", response_model=DeleteResponse, summary="Eliminar perfil academico")
def remove_profile():
    try:
        deleted = delete_profile()
    except Exception as exc:
        logger.exception("Error eliminando perfil: %s", exc)
        raise HTTPException(status_code=503, detail="No se pudo eliminar el perfil.") from exc

    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Perfil no encontrado.")

    return {"deleted": True, "message": "Perfil eliminado correctamente."}
