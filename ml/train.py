import os
import json
import subprocess
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

load_dotenv()

DATA_PATH = "data/raw/credit_default.csv"
MODEL_NAME = "credit-default-model"

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def get_dvc_data_version():
    try:
        with open(f"{DATA_PATH}.dvc") as f:
            content = yaml.safe_load(f)
            return content["outs"][0]["md5"]
    except Exception:
        return "unknown"

def main():
    with open("ml/params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["default_next_month"])
    y = df["default_next_month"]

    # Ordre des features, figé au moment de l'entraînement
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"],
        random_state=params["random_state"], stratify=y
    )

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("credit-default")

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("dvc_data_version", get_dvc_data_version())

        # Logger la liste ordonnée des features comme artefact
        with open("feature_names.json", "w") as f:
            json.dump(feature_names, f)
        mlflow.log_artifact("feature_names.json")

        # Enregistrer le modèle dans le Registry
        mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name=MODEL_NAME
        )

        # Nettoyage du fichier temporaire local
        os.remove("feature_names.json")

        print(f"Run ID: {run.info.run_id}")
        print(f"Metrics: {metrics}")
        print("Modèle + feature_names enregistrés dans MLflow.")

if __name__ == "__main__":
    main()