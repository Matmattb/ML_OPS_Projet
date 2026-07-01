import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = "data/raw/credit_default.csv"
MODEL_PATH = "models/model.joblib"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["default_next_month"])
    y = df["default_next_month"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")

if __name__ == "__main__":
    main()