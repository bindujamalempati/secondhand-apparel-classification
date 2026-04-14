from __future__ import annotations

import shutil
from pathlib import Path

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.config import MODELS_DIR, PROJECT_ROOT
from src.features import extract_features

app = FastAPI(title="Secondhand Apparel Classification API", version="1.0.0")

MODEL_PATH = MODELS_DIR / "apparel_classifier.joblib"
TEMP_DIR = PROJECT_ROOT / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

if MODEL_PATH.exists():
    MODEL_BUNDLE = joblib.load(MODEL_PATH)
else:
    MODEL_BUNDLE = None


@app.get("/")
def healthcheck():
    return {
        "status": "ok",
        "model_loaded": MODEL_BUNDLE is not None,
        "message": "Run prepare_data.py and train.py first if model_loaded is false.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=400, detail="Model not found. Run train.py first.")

    suffix = Path(file.filename).suffix or ".jpg"
    temp_path = TEMP_DIR / f"upload{suffix}"
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    features = extract_features(temp_path).reshape(1, -1)
    pipeline = MODEL_BUNDLE["pipeline"]
    pred = pipeline.predict(features)[0]
    probs = pipeline.predict_proba(features)[0]
    classes = pipeline.named_steps["clf"].classes_
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    temp_path.unlink(missing_ok=True)

    return {
        "predicted_label": pred,
        "top_predictions": [
            {"label": label, "probability": round(float(probability), 4)}
            for label, probability in ranked[:3]
        ],
    }
