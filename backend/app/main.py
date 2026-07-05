import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import get_model, load_model, get_feature_names
from app.schemas import PredictionInput, PredictionOutput
import time
from fastapi import Response
from app.monitoring import (
    prediction_requests_total,
    prediction_latency_seconds,
    prediction_failures_total,
    get_metrics,
)
import os
from fastapi import Request, HTTPException

METRICS_TOKEN = os.environ.get("METRICS_TOKEN", "")

def verify_metrics_token(request: Request):
    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {METRICS_TOKEN}"
    if not METRICS_TOKEN or auth != expected:
        raise HTTPException(status_code=401, detail="Non autorisé")

app = FastAPI(title="Credit Default API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    try:
        load_model()
    except Exception as e:
        print(f"Attention : modèle non chargé au démarrage : {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    prediction_requests_total.inc()  # compteur de requêtes
    start = time.time()
    try:
        model = get_model()
        feature_names = get_feature_names()
        df = pd.DataFrame([payload.features])

        missing = set(feature_names) - set(df.columns)
        if missing:
            prediction_failures_total.inc()  # échec
            raise HTTPException(
                status_code=422,
                detail=f"Features manquantes : {sorted(missing)}"
            )

        df = df[feature_names]
        pred = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1])

        prediction_latency_seconds.observe(time.time() - start)  # latence
        return PredictionOutput(prediction=pred, probability=proba)
    except HTTPException:
        raise
    except Exception as e:
        prediction_failures_total.inc()  # échec
        raise HTTPException(status_code=500, detail=str(e))
    
from fastapi import Depends

@app.get("/metrics")
def metrics(_: None = Depends(verify_metrics_token)):
    data, content_type = get_metrics()
    return Response(content=data, media_type=content_type)