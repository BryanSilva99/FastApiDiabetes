from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import database
from api.main import app
from api.services.model_service import model_service


VALID_PAYLOAD = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 79,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 35,
}


@pytest.fixture(autouse=True)
def temp_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "predictions_test.db"
    monkeypatch.setattr(database, "get_db_path", lambda: db_path)
    database.init_database()
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_health_responds(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["api_version"]
    assert data["feature_order"] == model_service.feature_order


def test_predict_accepts_valid_payload(client: TestClient):
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert data["id"] > 0
    assert data["prediction"] in [0, 1]
    assert data["risk"] in ["bajo", "medio", "alto"]
    assert data["model_name"]
    assert data["model_version"]


def test_predict_rejects_glucose_out_of_range(client: TestClient):
    payload = {**VALID_PAYLOAD, "Glucose": 0}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_invalid_age(client: TestClient):
    payload = {**VALID_PAYLOAD, "Age": 0}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_returns_valid_probability(client: TestClient):
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()

    assert 0 <= data["probability"] <= 1
    assert 0 <= data["risk_percentage"] <= 100
    assert data["risk_percentage"] == round(data["probability"] * 100, 2)


def test_prediction_is_saved(client: TestClient):
    client.post("/predict", json=VALID_PAYLOAD)
    response = client.get("/predictions")

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1


def test_prediction_detail_returns_saved_record(client: TestClient):
    create_response = client.post("/predict", json=VALID_PAYLOAD)
    prediction_id = create_response.json()["id"]

    response = client.get(f"/predictions/{prediction_id}")

    assert response.status_code == 200
    assert response.json()["id"] == prediction_id
    assert response.json()["glucose"] == VALID_PAYLOAD["Glucose"]


def test_create_prediction_endpoint_alias(client: TestClient):
    response = client.post("/predictions", json=VALID_PAYLOAD)

    assert response.status_code == 201
    assert response.json()["id"] > 0


def test_update_prediction_recalculates_and_preserves_id(client: TestClient):
    create_response = client.post("/predict", json=VALID_PAYLOAD)
    prediction_id = create_response.json()["id"]
    payload = {**VALID_PAYLOAD, "Glucose": 140, "BMI": 30.1}

    response = client.put(f"/predictions/{prediction_id}", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == prediction_id
    assert data["glucose"] == 140
    assert data["bmi"] == 30.1
    assert "updated_at" in data


def test_update_prediction_not_found_returns_404(client: TestClient):
    response = client.put("/predictions/999999", json=VALID_PAYLOAD)

    assert response.status_code == 404


def test_delete_one_prediction(client: TestClient):
    create_response = client.post("/predict", json=VALID_PAYLOAD)
    prediction_id = create_response.json()["id"]

    delete_response = client.delete(f"/predictions/{prediction_id}")
    detail_response = client.get(f"/predictions/{prediction_id}")

    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True
    assert detail_response.status_code == 404


def test_delete_prediction_not_found_returns_404(client: TestClient):
    response = client.delete("/predictions/999999")

    assert response.status_code == 404


def test_predictions_respects_limit(client: TestClient):
    client.post("/predict", json=VALID_PAYLOAD)
    client.post("/predict", json={**VALID_PAYLOAD, "Age": 55})

    response = client.get("/predictions?limit=1")

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1


def test_predictions_limit_too_high_returns_422(client: TestClient):
    response = client.get("/predictions?limit=101")

    assert response.status_code == 422


def test_delete_predictions(client: TestClient):
    client.post("/predict", json=VALID_PAYLOAD)

    delete_response = client.delete("/predictions")
    list_response = client.get("/predictions")

    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] == 1
    assert list_response.json()["predictions"] == []


def test_model_unavailable_returns_503(client: TestClient):
    original_model = model_service.model
    model_service.model = None
    try:
        response = client.post("/predict", json=VALID_PAYLOAD)
    finally:
        model_service.model = original_model

    assert response.status_code == 503


def test_create_preferences(client: TestClient):
    response = client.post(
        "/preferences",
        json={"theme": "light", "text_size": "normal"},
    )

    assert response.status_code == 201
    assert response.json()["theme"] == "light"
    assert response.json()["text_size"] == "normal"


def test_get_preferences(client: TestClient):
    client.post(
        "/preferences",
        json={"theme": "light", "text_size": "normal"},
    )

    response = client.get("/preferences")

    assert response.status_code == 200
    assert response.json()["theme"] == "light"


def test_get_preferences_not_found_returns_404(client: TestClient):
    response = client.get("/preferences")

    assert response.status_code == 404


def test_update_preferences(client: TestClient):
    client.post(
        "/preferences",
        json={"theme": "light", "text_size": "normal"},
    )

    response = client.put("/preferences", json={"theme": "dark", "text_size": "large"})

    assert response.status_code == 200
    assert response.json()["theme"] == "dark"
    assert response.json()["text_size"] == "large"


def test_update_preferences_not_found_returns_404(client: TestClient):
    response = client.put("/preferences", json={"theme": "dark"})

    assert response.status_code == 404


def test_delete_preferences(client: TestClient):
    client.post(
        "/preferences",
        json={"theme": "dark", "text_size": "large"},
    )

    delete_response = client.delete("/preferences")
    get_response = client.get("/preferences")

    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True
    assert get_response.status_code == 404


def test_preferences_validation_returns_422(client: TestClient):
    response = client.post("/preferences", json={"theme": "system", "text_size": "huge"})

    assert response.status_code == 422
