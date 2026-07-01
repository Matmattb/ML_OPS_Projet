from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np

# On patche le chargement du modèle AVANT d'importer l'app
def make_fake_model():
    fake = MagicMock()
    fake.predict.return_value = np.array([1])
    fake.predict_proba.return_value = np.array([[0.3, 0.7]])
    return fake

FAKE_FEATURES = ["LIMIT_BAL", "AGE"]

def test_health_endpoint():
    """La route /health répond ok."""
    with patch("app.model.load_model"), \
         patch("app.model.get_model", return_value=make_fake_model()), \
         patch("app.model.get_feature_names", return_value=FAKE_FEATURES):
        from app.main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

def test_predict_endpoint():
    """La route /predict renvoie une prédiction structurée."""
    with patch("app.model.load_model"), \
         patch("app.main.get_model", return_value=make_fake_model()), \
         patch("app.main.get_feature_names", return_value=FAKE_FEATURES):
        from app.main import app
        client = TestClient(app)
        payload = {"features": {"LIMIT_BAL": 20000, "AGE": 24}}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["prediction"] == 1
        assert 0 <= body["probability"] <= 1