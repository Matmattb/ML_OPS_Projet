import os
import json
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "credit-default-model"
MODEL_STAGE = os.environ.get("MODEL_STAGE", "None")

_model = None
_feature_names = None

def _resolve_version():
    """Trouve la version du modèle à charger (par stage, ou la dernière)."""
    client = MlflowClient()
    if MODEL_STAGE and MODEL_STAGE != "None":
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    else:
        versions = client.get_latest_versions(MODEL_NAME)
    if not versions:
        raise RuntimeError(f"Aucune version trouvée pour {MODEL_NAME} (stage={MODEL_STAGE})")
    # On prend la version au numéro le plus élevé
    return max(versions, key=lambda v: int(v.version))

def load_model():
    global _model, _feature_names
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    mv = _resolve_version()

    # 1. Charger le modèle
    _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")

    # 2. Charger feature_names.json depuis le run qui a produit ce modèle
    client = MlflowClient()
    local_path = client.download_artifacts(mv.run_id, "feature_names.json")
    with open(local_path) as f:
        _feature_names = json.load(f)

    print(f"Modèle v{mv.version} chargé avec {len(_feature_names)} features.")
    return _model

def get_model():
    global _model
    if _model is None:
        load_model()
    return _model

def get_feature_names():
    global _feature_names
    if _feature_names is None:
        load_model()
    return _feature_names