from typing import Literal
from pydantic import BaseModel, Field


RiskLevel = Literal["bajo", "medio", "alto"]
ThemePreference = Literal["system", "light", "dark"]


class DiabetesInput(BaseModel):
    Pregnancies: int = Field(ge=0, le=20, description="Numero de embarazos")
    Glucose: float = Field(gt=0, le=250, description="Glucosa en mg/dL")
    BloodPressure: float = Field(gt=0, le=150, description="Presion arterial en mmHg")
    SkinThickness: float = Field(ge=0, le=100, description="Grosor de piel en mm")
    Insulin: float = Field(ge=0, le=900, description="Nivel de insulina")
    BMI: float = Field(gt=0, le=80, description="Indice de masa corporal")
    DiabetesPedigreeFunction: float = Field(ge=0, le=3, description="Funcion pedigree de diabetes")
    Age: int = Field(ge=1, le=120, description="Edad")


class PredictionResponse(BaseModel):
    id: int
    prediction: int = Field(ge=0, le=1)
    risk: RiskLevel
    probability: float = Field(ge=0, le=1, description="Probabilidad estimada por el modelo para la clase positiva")
    risk_percentage: float = Field(ge=0, le=100, description="Porcentaje estimado derivado de probability")
    threshold: float = Field(ge=0, le=1)
    message: str
    recommendation: str
    model_name: str
    model_version: str
    created_at: str
    updated_at: str | None = None


class PredictionHistoryItem(PredictionResponse):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int


class PredictionsResponse(BaseModel):
    predictions: list[PredictionHistoryItem]


class DeleteHistoryResponse(BaseModel):
    deleted: int
    message: str


class DeleteResponse(BaseModel):
    deleted: bool
    message: str


class ProfileBase(BaseModel):
    alias: str | None = Field(default=None, max_length=40, description="Alias opcional sin datos sensibles")
    age: int | None = Field(default=None, ge=1, le=120, description="Edad general opcional")
    preferred_theme: ThemePreference = Field(default="system", description="Preferencia visual")
    text_scale: float = Field(default=1.0, ge=0.8, le=1.4, description="Escala de texto preferida")


class ProfileCreateRequest(ProfileBase):
    pass


class ProfileUpdateRequest(BaseModel):
    alias: str | None = Field(default=None, max_length=40)
    age: int | None = Field(default=None, ge=1, le=120)
    preferred_theme: ThemePreference | None = None
    text_scale: float | None = Field(default=None, ge=0.8, le=1.4)


class ProfileResponse(ProfileBase):
    id: int
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api_version: str
    model_loaded: bool
    model_name: str | None
    model_version: str | None
    database_available: bool
    feature_order: list[str]


class MetricsResponse(BaseModel):
    metrics: list[dict[str, float | str]]
    message: str | None = None


class RootResponse(BaseModel):
    message: str
