from __future__ import annotations

import argparse
import json

import joblib

from src.config import MODELS_DIR
from src.features import extract_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a single apparel image.")
    parser.add_argument("image_path", type=str)
    args = parser.parse_args()

    model_bundle = joblib.load(MODELS_DIR / "apparel_classifier.joblib")
    pipeline = model_bundle["pipeline"]
    features = extract_features(args.image_path).reshape(1, -1)
    pred = pipeline.predict(features)[0]
    probs = pipeline.predict_proba(features)[0]
    classes = pipeline.named_steps["clf"].classes_
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

    result = {
        "predicted_label": pred,
        "top_predictions": [{"label": c, "probability": round(float(p), 4)} for c, p in ranked[:3]],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
