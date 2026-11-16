"""Train final teen-stress pipeline. Mirrors notebooks/notebook_main.ipynb Sec 5/6/9."""
import argparse
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

RANDOM_STATE = 42
TARGET = "survey_stress_score"
NUM_FEATURES = ["age", "social_media_hours", "exercise_hours", "sleep_hours", "screen_time_hours"]
CAT_FEATURES = ["gender", "support_system", "academic_performance"]

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mental_health_analysis.csv"
MODEL_PATH = ROOT / "models" / "final_pipeline.joblib"
META_PATH = ROOT / "models" / "final_metadata.joblib"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    X = df.drop(columns=["user_id", "wearable_stress_score", TARGET])
    y = (df[TARGET] - 1).astype(int)  # 1-5 -> 0-4
    return X, y


def build_pipeline():
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    return Pipeline([("prep", prep), ("clf", clf)])


def main(data_path=DATA_PATH, model_path=MODEL_PATH, meta_path=META_PATH):
    X, y = load_data(data_path)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25,
        stratify=y_train_full, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train)} Val: {len(X_val)} Test: {len(X_test)}")

    pipe = build_pipeline()
    pipe.fit(X_train_full, y_train_full)

    pred = pipe.predict(X_test)
    metrics = {
        "acc": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "n_test": int(len(X_test)),
    }
    print(f"Test acc={metrics['acc']} macro_f1={metrics['macro_f1']}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    joblib.dump({"metrics": metrics}, meta_path)
    print(f"Saved {model_path} + {meta_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--meta", default=str(META_PATH))
    a = p.parse_args()
    main(Path(a.data), Path(a.model), Path(a.meta))
