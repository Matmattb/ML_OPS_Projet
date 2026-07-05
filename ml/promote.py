import os
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "credit-default-model"

def main():
    with open("candidate_version.txt") as f:
        version = f.read().strip()

    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])

    # Attribuer l'alias 'production' à la version validée
    client.set_registered_model_alias(MODEL_NAME, "production", version)
    print(f" Version {version} promue en Production (alias 'production').")

if __name__ == "__main__":
    main()