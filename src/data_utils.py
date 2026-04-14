from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TARGET_CLASSES = ["Tshirts", "Shirts", "Jeans", "Dresses"]
RANDOM_STATE = 42


def reset_dirs() -> None:
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def split_dataset(df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    train_records = []
    val_records = []
    test_records = []

    for label in TARGET_CLASSES:
        label_df = df[df["label"] == label].sample(frac=1.0, random_state=seed).reset_index(drop=True)

        n = len(label_df)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        train_df = label_df.iloc[:n_train]
        val_df = label_df.iloc[n_train:n_train + n_val]
        test_df = label_df.iloc[n_train + n_val:]

        for split_name, split_df, bucket in [
            ("train", train_df, train_records),
            ("val", val_df, val_records),
            ("test", test_df, test_records),
        ]:
            out_dir = PROCESSED_DIR / split_name / label
            out_dir.mkdir(parents=True, exist_ok=True)

            for _, row in split_df.iterrows():
                src = Path(row["image_path"])
                dst = out_dir / src.name
                shutil.copy2(src, dst)

                record = row.to_dict()
                record["split"] = split_name
                record["processed_image_path"] = str(dst)
                bucket.append(record)

    all_records = train_records + val_records + test_records
    return pd.DataFrame(all_records)


def download_and_prepare(limit_per_class: int = 250) -> pd.DataFrame:
    reset_dirs()

    dataset = load_dataset("ashraq/fashion-product-images-small", split="train")

    counts = defaultdict(int)
    records = []

    for row in dataset:
        master_category = str(row.get("masterCategory", "")).strip()
        sub_category = str(row.get("subCategory", "")).strip()
        article_type = str(row.get("articleType", "")).strip()
        gender = str(row.get("gender", "")).strip()
        image = row.get("image", None)
        image_id = row.get("id", len(records))

        if master_category != "Apparel":
            continue
        if article_type not in TARGET_CLASSES:
            continue
        if image is None:
            continue
        if counts[article_type] >= limit_per_class:
            continue

        class_dir = RAW_DIR / article_type
        class_dir.mkdir(parents=True, exist_ok=True)

        image_path = class_dir / f"img_{int(image_id):05d}.jpg"
        image.convert("RGB").save(image_path)

        records.append(
            {
                "image_path": str(image_path),
                "label": article_type,
                "master_category": master_category,
                "sub_category": sub_category,
                "article_type": article_type,
                "gender": gender,
            }
        )
        counts[article_type] += 1

        if all(counts[c] >= limit_per_class for c in TARGET_CLASSES):
            break

    if not records:
        raise RuntimeError("No records were downloaded. Check dataset availability or class names.")

    raw_df = pd.DataFrame(records).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    final_df = split_dataset(raw_df, seed=RANDOM_STATE)

    metadata_path = PROCESSED_DIR / "metadata.csv"
    final_df.to_csv(metadata_path, index=False)

    stats_path = PROCESSED_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "counts_per_class": {k: int(v) for k, v in counts.items()},
                "total_raw_records": int(len(raw_df)),
                "total_processed_records": int(len(final_df)),
                "classes": TARGET_CLASSES,
            },
            f,
            indent=2,
        )

    return final_df