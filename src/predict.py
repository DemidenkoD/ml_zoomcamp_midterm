"""Load final pipeline and predict a single teen profile."""
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_pipeline.joblib"

FEATURES = ["age", "social_media_hours", "exercise_hours", "sleep_hours",
            "screen_time_hours", "gender", "support_system", "academic_performance"]
LABELS = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

_pipe = None


def get_pipeline(path=MODEL_PATH):
    global _pipe
    if _pipe is None:
        _pipe = joblib.load(path)
    return _pipe


def predict_stress(profile: dict, path=MODEL_PATH):
    """profile: dict with the 8 raw features. Returns (level_1_to_5, proba_list)."""
    pipe = get_pipeline(path)
    row = pd.DataFrame([{k: profile[k] for k in FEATURES}])
    proba = pipe.predict_proba(row)[0].tolist()
    level = LABELS[int(pipe.predict(row)[0])]
    return level, proba


if __name__ == "__main__":
    sample = {"age": 16, "social_media_hours": 5.0, "exercise_hours": 1.5,
              "sleep_hours": 7.0, "screen_time_hours": 7.0,
              "gender": "F", "support_system": "Moderate",
              "academic_performance": "Good"}
    level, proba = predict_stress(sample)
    print(f"Predicted level: {level}")
    print(f"Probas: {[round(p, 3) for p in proba]}")
