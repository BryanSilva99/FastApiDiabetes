from fastapi import FastAPI
import joblib
import numpy as np
import os

app = FastAPI()

# ruta correcta del modelo
base_path = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_path, "model", "model.pkl")

model = joblib.load(model_path)


@app.get("/")
def home():
    return {"message": "API Diabetes funcionando"}


@app.post("/predict")
def predict(data: dict):
    values = np.array(list(data.values())).reshape(1, -1)
    prediction = model.predict(values)[0]

    return {
        "prediction": int(prediction),
        "message": "Alto riesgo de diabetes" if prediction == 1 else "Bajo riesgo"
    }
