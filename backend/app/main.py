import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import get_model, load_model, get_feature_names
from app.schemas import PredictionInput, PredictionOutput

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
    try:
        model = get_model()
        feature_names = get_feature_names()
        df = pd.DataFrame([payload.features])

        missing = set(feature_names) - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Features manquantes : {sorted(missing)}"
            )
        df = df[feature_names]

        pred = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1])
        return PredictionOutput(prediction=pred, probability=proba)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))