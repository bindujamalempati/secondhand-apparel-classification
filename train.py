from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR, RANDOM_STATE
from src.features import extract_features


def main() -> None:
    metadata_path = PROCESSED_DIR / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError("Run prepare_data.py first to create metadata.csv")

    df = pd.read_csv(metadata_path)

    image_col = "processed_image_path" if "processed_image_path" in df.columns else "image_path"

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    X_train = [extract_features(path) for path in train_df[image_col]]
    y_train = train_df["label"].tolist()

    X_test = [extract_features(path) for path in test_df[image_col]]
    y_test = test_df["label"].tolist()

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=18,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    classes = pipeline.named_steps["clf"].classes_.tolist()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "apparel_classifier.joblib"
    joblib.dump({"pipeline": pipeline, "classes": classes}, model_path)

    metrics = {
        "accuracy": acc,
        "num_samples": len(df),
        "classes": classes,
        "classification_report": report,
    }
    with open(OUTPUTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, preds, labels=classes)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.title(f"Confusion Matrix | accuracy={acc:.3f}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)

    preview = pd.DataFrame(
        {
            "true_label": y_test[:20],
            "predicted_label": preds[:20],
            "top_probability": [float(max(p)) for p in proba[:20]],
        }
    )
    preview.to_csv(OUTPUTS_DIR / "prediction_preview.csv", index=False)

    print(f"Training complete. Accuracy: {acc:.4f}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()