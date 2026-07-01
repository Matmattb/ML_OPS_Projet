import pandas as pd
from app.schemas import PredictionInput, PredictionOutput

def test_prediction_input_valid():
    """Un input bien formé est accepté et parsé."""
    data = {"features": {"LIMIT_BAL": 20000, "AGE": 24}}
    obj = PredictionInput(**data)
    assert obj.features["LIMIT_BAL"] == 20000
    assert obj.features["AGE"] == 24

def test_prediction_output_types():
    """La sortie force bien les bons types."""
    out = PredictionOutput(prediction=1, probability=0.73)
    assert isinstance(out.prediction, int)
    assert isinstance(out.probability, float)
    assert out.prediction == 1

def test_feature_reordering_logic():
    """Réordonner un DataFrame selon une liste de features fonctionne."""
    feature_names = ["A", "B", "C"]
    df = pd.DataFrame([{"C": 3, "A": 1, "B": 2}])  # ordre mélangé
    df_ordered = df[feature_names]
    assert list(df_ordered.columns) == ["A", "B", "C"]
    assert df_ordered.iloc[0]["A"] == 1