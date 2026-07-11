# Diabetes Risk API

Backend academico desarrollado con FastAPI, SQLite y scikit-learn para estimar de forma referencial el riesgo de diabetes.

> Advertencia medica: esta API no entrega diagnosticos medicos ni reemplaza una evaluacion profesional.

## Nombre Real De Carpeta

En el workspace local, el backend esta en:

```text
/home/bryan/Documentos/Escritorio/SOFTMOVIL/FastApi
```

El remoto configurado apunta a `BryanSilva99/FastApiDiabetes.git`.

## Tecnologias

- FastAPI.
- Pydantic.
- SQLite.
- pandas.
- scikit-learn.
- joblib.
- pytest.
- Uvicorn.

## Estructura

```text
api/
  config.py              configuracion por variables de entorno
  database.py            SQLite, migraciones simples y CRUD
  main.py                endpoints FastAPI
  schemas.py             modelos Pydantic
  services/
    model_service.py     carga del modelo y prediccion
model/
  model.pkl              pipeline entrenado
  model_metadata.json    metadata del modelo
  metrics.csv/json       metricas del entrenamiento
  confusion_matrix.json
  roc_curve.csv
  train.py
data/
  diabetes.csv           dataset
tests/
  test_api.py            pruebas con TestClient y base temporal
```

## Variables Del Modelo

Orden obligatorio:

```text
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
```

El backend carga este orden desde `model/model_metadata.json`.

## Instalacion

Linux:

```bash
cd /home/bryan/Documentos/Escritorio/SOFTMOVIL/FastApi
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd C:\ruta\al\proyecto\FastApi
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows CMD:

```bat
cd C:\ruta\al\proyecto\FastApi
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Variables De Entorno

Copiar `.env.example` a `.env` si se quiere personalizar:

```env
API_TITLE=Diabetes Risk API
API_VERSION=1.1.0
CORS_ORIGINS=*
MODEL_PATH=model/model.pkl
METADATA_PATH=model/model_metadata.json
METRICS_PATH=model/metrics.csv
DATABASE_PATH=data/predictions.db
HISTORY_DEFAULT_LIMIT=20
HISTORY_MAX_LIMIT=100
```

`CORS_ORIGINS=*` es aceptable para desarrollo local. En despliegue debe reemplazarse por origenes concretos.

## Ejecucion Local

Con entorno activado:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger:

```text
http://localhost:8000/docs
```

Health:

```text
http://localhost:8000/health
```

## SQLite

La base por defecto se ubica en:

```text
data/predictions.db
```

Esta base no debe versionarse. Se ignora mediante `.gitignore`.

Tablas:

- `predictions`: evaluaciones y resultados calculados.
- `preferences`: tema y tamano de texto de la instalacion anonima.

## Metadata Del Modelo

Archivo:

```text
model/model_metadata.json
```

Contiene nombre, version, orden de caracteristicas, umbral, metricas principales y decisiones de entrenamiento.

## Entrenamiento

```bash
python model/train.py
```

Genera o actualiza:

- `model/model.pkl`
- `model/model_metadata.json`
- `model/metrics.csv`
- `model/metrics.json`
- `model/confusion_matrix.json`
- `model/roc_curve.csv`

Modelo documentado actualmente:

```text
Arbol de Decision v1.0.0
```

Metricas registradas:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.7597 | 0.6393 | 0.7222 | 0.6783 | 0.7622 |

## Endpoints

| Metodo | Endpoint | Descripcion |
| --- | --- | --- |
| `GET` | `/` | Estado basico. |
| `GET` | `/health` | Estado de API, modelo, base y orden de variables. |
| `GET` | `/model/metrics` | Metricas del entrenamiento. |
| `POST` | `/predict` | Crea evaluacion y calcula riesgo. |
| `POST` | `/predictions` | Crea evaluacion; alias explicito de creacion. |
| `GET` | `/predictions?limit=20` | Lista historial con limite validado. |
| `GET` | `/predictions/{id}` | Consulta detalle de evaluacion. |
| `PUT` | `/predictions/{id}` | Actualiza ocho valores y recalcula. |
| `DELETE` | `/predictions/{id}` | Elimina una evaluacion. |
| `DELETE` | `/predictions` | Elimina todo el historial. |
| `POST` | `/preferences` | Crea preferencias. |
| `GET` | `/preferences` | Consulta preferencias. |
| `PUT` | `/preferences` | Actualiza preferencias. |
| `DELETE` | `/preferences` | Restaura configuracion predeterminada. |

## CRUD De Evaluaciones

- Create: `POST /predict` o `POST /predictions`.
- Read: `GET /predictions`, `GET /predictions/{id}`.
- Update: `PUT /predictions/{id}`.
- Delete: `DELETE /predictions/{id}`, `DELETE /predictions`.

Al actualizar una evaluacion, el backend recalcula `prediction`, `risk`, `probability`, `risk_percentage`, `message` y `recommendation`. No se aceptan ediciones manuales de esos campos.

## CRUD De Preferencias

Preferencias anonimas de experiencia:

- `theme`: `light` o `dark`.
- `text_size`: `normal` o `large`.
- `created_at`.
- `updated_at`.

No guarda alias, edad, DNI, correo, telefono, direccion, credenciales ni datos clinicos adicionales.

## Ejemplo `/predict`

Request:

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

Response:

```json
{
  "id": 11,
  "prediction": 1,
  "risk": "alto",
  "probability": 0.5169491525423728,
  "risk_percentage": 51.69,
  "threshold": 0.5,
  "message": "Riesgo alto estimado por el modelo",
  "recommendation": "Consulta con un profesional de salud para una evaluacion preventiva.",
  "model_name": "Arbol de Decision",
  "model_version": "1.0.0",
  "created_at": "2026-07-10T14:35:27.378427+00:00",
  "updated_at": "2026-07-10T14:35:27.378427+00:00"
}
```

## Ejemplo `/preferences`

Request:

```json
{
  "theme": "dark",
  "text_size": "large"
}
```

## Docker

Existe `Dockerfile`.

Construir:

```bash
docker build -t diabetes-api .
```

Ejecutar con persistencia de SQLite:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" diabetes-api
```

En Windows PowerShell:

```powershell
docker run --rm -p 8000:8000 -v "${PWD}\data:/app/data" diabetes-api
```

## Despliegue

Para VPS se recomienda:

- usar entorno virtual o Docker;
- exponer Uvicorn detras de Nginx;
- configurar HTTPS;
- definir `CORS_ORIGINS` con el dominio real;
- persistir `data/predictions.db`.

## Pruebas

Con entorno activado:

```bash
pytest -q
```

Resultado real ejecutado:

```text
23 passed, 1 warning
```

El warning actual proviene de Starlette/TestClient recomendando `httpx2`.

## Limitaciones

- API academica sin autenticacion.
- No administra historias clinicas.
- SQLite se usa con fines academicos.
- La estimacion depende del dataset y modelo entrenado.
- No reemplaza criterio medico profesional.
