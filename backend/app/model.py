import os
import json
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "credit-default-model"
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")

_model = None
_feature_names = None

def load_model():
    global _model, _feature_names
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

    _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    local_path = client.download_artifacts(mv.run_id, "feature_names.json")
    with open(local_path) as f:
        _feature_names = json.load(f)

    print(f"Modèle v{mv.version} (alias '{MODEL_ALIAS}') chargé avec {len(_feature_names)} features.")
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