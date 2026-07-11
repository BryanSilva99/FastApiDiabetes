# Diabetes Risk API

Backend academico con FastAPI, SQLite y scikit-learn para estimar de forma referencial el riesgo de diabetes.

> No es un diagnostico medico ni reemplaza una evaluacion profesional.

## Tecnologias

- FastAPI
- Pydantic
- SQLite
- pandas
- scikit-learn
- pytest

## Variables del Modelo

Orden obligatorio:

`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.

## Instalacion

Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Entrenamiento

```bash
python model/train.py
```

Genera:

- `model/model.pkl`
- `model/model_metadata.json`
- `model/metrics.csv`
- `model/metrics.json`
- `model/confusion_matrix.json`
- `model/roc_curve.csv`

Modelo seleccionado: `Arbol de Decision` v`1.0.0`.

| Accuracy | Precision | Recall | F1 | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.7597 | 0.6393 | 0.7222 | 0.6783 | 0.7622 |

## Ejecucion

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger:

```text
http://localhost:8000/docs
```

## Variables de Entorno

Copiar `.env.example` a `.env` si se quiere personalizar:

```env
CORS_ORIGINS=*
DATABASE_PATH=data/predictions.db
MODEL_PATH=model/model.pkl
METADATA_PATH=model/model_metadata.json
```

`CORS_ORIGINS=*` queda permitido solo para desarrollo local. En despliegue usar dominios concretos.

## Docker

```bash
docker build -t diabetes-api .
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" diabetes-api
```

## Endpoints

| Metodo | Endpoint | Descripcion |
| --- | --- | --- |
| `GET` | `/` | Estado basico. |
| `GET` | `/health` | API, modelo, base de datos y orden de variables. |
| `GET` | `/model/metrics` | Metricas generadas por entrenamiento. |
| `POST` | `/predict` | Prediccion y guardado en historial. |
| `POST` | `/predictions` | Crea una evaluacion. Alternativa explicita a `/predict`. |
| `GET` | `/predictions?limit=20` | Historial con limite validado. |
| `GET` | `/predictions/{id}` | Detalle de una evaluacion. |
| `PUT` | `/predictions/{id}` | Edita los ocho valores y recalcula la evaluacion. |
| `DELETE` | `/predictions/{id}` | Borra una evaluacion. |
| `DELETE` | `/predictions` | Borra todo el historial. |
| `POST` | `/profile` | Crea el perfil academico de usuario unico. |
| `GET` | `/profile` | Consulta el perfil. |
| `PUT` | `/profile` | Actualiza alias, edad o preferencias. |
| `DELETE` | `/profile` | Elimina el perfil. |

## CRUD Academico

La entrega expone dos entidades persistidas en SQLite:

1. **Evaluaciones**: cada prediccion guardada. `POST /predict` o `POST /predictions` crea, `GET /predictions` lista, `GET /predictions/{id}` consulta detalle, `PUT /predictions/{id}` actualiza los ocho indicadores y recalcula el resultado con el modelo, y `DELETE` elimina uno o todos los registros.
2. **Perfil**: perfil academico minimo de usuario unico. Incluye `alias` opcional, `age` opcional, `preferred_theme`, `text_scale`, `created_at` y `updated_at`. No guarda DNI, correo, telefono, direccion, credenciales ni historias clinicas.

Al editar una evaluacion no se permite modificar manualmente `prediction`, `risk`, `probability`, `message` ni `recommendation`; esos campos se recalculan desde el pipeline cargado.

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
  "recommendation": "Se recomienda solicitar orientacion profesional y revisar habitos de alimentacion, actividad fisica y controles preventivos.",
  "model_name": "Arbol de Decision",
  "model_version": "1.0.0",
  "created_at": "2026-07-10T14:35:27.378427+00:00",
  "updated_at": "2026-07-10T14:35:27.378427+00:00"
}
```

## Ejemplo `/profile`

Request:

```json
{
  "alias": "Estudiante",
  "age": 25,
  "preferred_theme": "system",
  "text_scale": 1.0
}
```

Response:

```json
{
  "id": 1,
  "alias": "Estudiante",
  "age": 25,
  "preferred_theme": "system",
  "text_scale": 1.0,
  "created_at": "2026-07-11T10:00:00+00:00",
  "updated_at": "2026-07-11T10:00:00+00:00"
}
```

## Historial y Privacidad

SQLite se usa con fines academicos. No se almacenan DNI, correo, telefono, direccion, credenciales ni identificadores personales fuertes. El endpoint `DELETE /predictions` elimina el historial completo y `DELETE /profile` elimina el perfil academico.

## Pruebas

```bash
pytest -q
```

Resultado verificado:

```text
23 passed, 1 warning
```
