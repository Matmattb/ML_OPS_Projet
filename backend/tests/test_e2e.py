import os
import pytest
from fastapi.testclient import TestClient

# Ce test nécessite un vrai accès MLflow. On le skip s'il n'est pas configuré.
requires_mlflow = pytest.mark.skipif(
    not os.environ.get("MLFLOW_TRACKING_URI"),
    reason="MLFLOW_TRACKING_URI non configuré (test E2E ignoré)"
)

@requires_mlflow
def test_full_prediction_flow():
    """Parcours complet : vrai modèle chargé depuis le registry -> vraie prédiction."""
    from app.main import app
    client = TestClient(app)

    payload = {"features": {
        "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1,
        "AGE": 24, "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1,
        "PAY_5": -2, "PAY_6": -2, "BILL_AMT1": 3913, "BILL_AMT2": 3102,
        "BILL_AMT3": 689, "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
        "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0,
        "PAY_AMT5": 0, "PAY_AMT6": 0
    }}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0