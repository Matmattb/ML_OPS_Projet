import os
import sys
import json
import yaml
import time
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

load_dotenv()

DATA_PATH = "data/raw/credit_default.csv"
MODEL_NAME = "credit-default-model"

GATES = {
    "min_accuracy": 0.78,
    "min_f1": 0.40,
    "max_latency_ms": 100, 
}

def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    candidate = max(versions, key=lambda v: int(v.version))
    print(f"Candidat évalué : version {candidate.version}")


    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{candidate.version}")


    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["default_next_month"])
    y = df["default_next_month"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    single = X_test.iloc[[0]]
    start = time.time()
    model.predict(single)
    latency_ms = (time.time() - start) * 1000

    print(f"Accuracy : {acc:.4f} (seuil {GATES['min_accuracy']})")
    print(f"F1       : {f1:.4f} (seuil {GATES['min_f1']})")
    print(f"Latence  : {latency_ms:.1f} ms (max {GATES['max_latency_ms']})")


    failures = []
    if acc < GATES["min_accuracy"]:
        failures.append(f"accuracy {acc:.4f} < {GATES['min_accuracy']}")
    if f1 < GATES["min_f1"]:
        failures.append(f"f1 {f1:.4f} < {GATES['min_f1']}")
    if latency_ms > GATES["max_latency_ms"]:
        failures.append(f"latence {latency_ms:.1f}ms > {GATES['max_latency_ms']}ms")

    if failures:
        print(" QUALITY GATES ÉCHOUÉES :")
        for f in failures:
            print(f"   - {f}")

        sys.exit(1) 
    else:
        print(" TOUTES LES GATES SONT PASSÉES")

        with open("candidate_version.txt", "w") as f:
            f.write(str(candidate.version))

if __name__ == "__main__":
    main()
