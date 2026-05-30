# Backend FastAPI - Diabetes App

Backend REST para ejecutar predicciones de riesgo de diabetes usando un modelo de Machine Learning entrenado con scikit-learn.

## Instalacion

```bash
cd FastApi
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows:

```bash
cd FastApi
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecucion

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

| Metodo | Endpoint | Descripcion |
| --- | --- | --- |
| `GET` | `/` | Mensaje de prueba de la API. |
| `GET` | `/health` | Estado de API, modelo, base de datos y variables. |
| `GET` | `/model/metrics` | Metricas de los modelos evaluados. |
| `POST` | `/predict` | Prediccion de riesgo y guardado en SQLite. |
| `GET` | `/predictions` | Historial de predicciones. |

## Entrenamiento del modelo

```bash
python model/train.py
```

El entrenamiento genera:

- `model/model.pkl`
- `model/metrics.csv`
- `model/metrics.json`

## Base de datos

La base SQLite se crea automaticamente en:

```text
FastApi/data/predictions.db
```

La tabla `predictions` guarda variables de entrada, resultado, recomendacion y fecha de creacion.

