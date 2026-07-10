# API

Base local sugerida: `http://IP_DE_TU_PC:8000`.

## Endpoints

| Metodo | Ruta | Descripcion |
| --- | --- | --- |
| `GET` | `/` | Estado basico. |
| `GET` | `/health` | Estado de API, modelo, base y variables. |
| `GET` | `/model/metrics` | Metricas reales generadas por entrenamiento. |
| `POST` | `/predict` | Estimacion referencial de riesgo. |
| `GET` | `/predictions?limit=20` | Historial paginado por limite. |
| `DELETE` | `/predictions` | Elimina todo el historial academico. |

## Request `/predict`

```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 79,
  "BMI": 28.5,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 35
}
```

## Response `/predict`

```json
{
  "id": 11,
  "prediction": 1,
  "risk": "alto",
  "probability": 0.5169491525423728,
  "risk_percentage": 51.69,
  "threshold": 0.5,
  "message": "Riesgo alto estimado por el modelo",
  "recommendation": "Se recomienda solicitar orientacion profesional y revisar habitos de alimentacion, actividad fisica y controles preventivos.",
  "model_name": "Arbol de Decision",
  "model_version": "1.0.0",
  "created_at": "2026-07-10T14:35:27.378427+00:00"
}
```

`probability` se obtiene de `predict_proba()` y `risk_percentage` es `probability * 100`, redondeado a dos decimales. No representa certeza clinica.

## Health

`GET /health` informa:

- `status`
- `api_version`
- `model_loaded`
- `model_name`
- `model_version`
- `database_available`
- `feature_order`

## Codigos relevantes

- `422`: validacion Pydantic fallida.
- `503`: modelo o base no disponible.
- `500`: error interno no esperado.
