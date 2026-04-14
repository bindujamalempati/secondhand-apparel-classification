# Secondhand Apparel Classification System Using Computer Vision

A lightweight computer vision portfolio project for **secondhand apparel classification** using a **real public fashion-product dataset**, handcrafted visual features, a **Random Forest classifier**, and a **FastAPI inference API**.

This project is designed as a practical, GitHub-ready demo of an apparel image classification workflow. It includes dataset preparation, train/validation/test split generation, model training, batch-free local inference, and an API endpoint for uploading an image and getting predicted apparel categories.

---

## Project Overview

The goal of this project is to classify secondhand apparel images into core clothing categories using a reproducible local pipeline.

This portfolio version uses a public apparel dataset as a proxy for resale marketplace inventory and demonstrates:

- real image dataset preparation
- preprocessing and dataset splitting
- handcrafted feature extraction
- supervised classification
- local prediction
- FastAPI-based inference service

### Supported Classes

The current version classifies images into these categories:

- `Shirts`
- `Tshirts`
- `Jeans`
- `Dresses`

---

## Why This Project

Secondhand and resale platforms often work with large volumes of product images that vary in:

- lighting
- pose
- cropping
- background clutter
- image quality

A lightweight visual classification system helps standardize product categorization and reduces manual effort in inventory listing workflows.

This project is a **portfolio implementation inspired by production-style apparel classification workflows**.

---

## Dataset

This project uses a **real public fashion-product image dataset** and prepares a filtered apparel subset locally.

### Source
- Public dataset: `ashraq/fashion-product-images-small`
- Accessed using the Hugging Face `datasets` library

### Dataset Handling in This Project
The project:
- downloads the dataset
- filters to apparel-only records
- selects target classes
- saves images locally
- creates train / val / test splits
- writes metadata to `data/processed/metadata.csv`

### Class Distribution
The current prepared dataset uses:

- 250 images per class
- 4 classes
- 1000 total source images

After splitting:
- 70% train
- 15% validation
- 15% test

---

## Model Approach

This implementation uses a **feature-based image classification pipeline** instead of a deep CNN.

### Pipeline
1. Load image
2. Resize and normalize
3. Extract handcrafted visual features
4. Train a `RandomForestClassifier`
5. Save trained model with `joblib`
6. Serve predictions through FastAPI

### Why This Approach
This version was intentionally kept lightweight so it can:
- run locally without GPU requirements
- train quickly on a laptop
- be easy to demonstrate and reproduce
- provide a complete end-to-end portfolio workflow

---

## Features

- Real public apparel dataset integration
- Automatic dataset preparation
- Balanced class sampling
- Train / validation / test split generation
- Metadata CSV creation
- Local training pipeline
- CLI prediction script
- FastAPI prediction endpoint
- Saved model artifact
- Metrics and confusion matrix output

---

## Project Structure

```text
apparel_cv_project/
│
├── app.py
├── train.py
├── predict.py
├── prepare_data.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train/
│       ├── val/
│       ├── test/
│       ├── metadata.csv
│       └── dataset_stats.json
│
├── models/
│   └── apparel_classifier.joblib
│
├── outputs/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── prediction_preview.csv
│
└── src/
    ├── config.py
    ├── data_utils.py
    ├── features.py
    └── ...
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/secondhand-apparel-classification-cv.git
cd secondhand-apparel-classification-cv
```

### 2. Create and activate a virtual environment

#### Windows CMD
```bat
python -m venv .venv
.venv\Scripts\activate
```

#### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install python-multipart
```

---

## How to Run

### Step 1: Prepare the dataset

```bash
python prepare_data.py --limit_per_class 250
```

This will:
- download the public dataset
- filter the target categories
- save local image files
- split into train / val / test
- generate metadata

---

### Step 2: Train the model

```bash
python train.py
```

This will:
- read `metadata.csv`
- extract features
- train the classifier
- save the trained model
- generate evaluation outputs

---

### Step 3: Run a prediction from command line

```bash
python predict.py "data\processed\test\Shirts\img_08181.jpg"
```

### Example Output

```json
{
  "predicted_label": "Shirts",
  "top_predictions": [
    {
      "label": "Shirts",
      "probability": 0.75
    },
    {
      "label": "Tshirts",
      "probability": 0.24
    },
    {
      "label": "Dresses",
      "probability": 0.01
    }
  ]
}
```

---

### Step 4: Start the FastAPI app

```bash
python -m uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Use the Swagger UI to test the `/predict` endpoint by uploading an image.

---

## API

### `POST /predict`

Upload an image file and receive predicted apparel category probabilities.

#### Response Example

```json
{
  "predicted_label": "Jeans",
  "top_predictions": [
    {
      "label": "Jeans",
      "probability": 0.81
    },
    {
      "label": "Tshirts",
      "probability": 0.11
    },
    {
      "label": "Shirts",
      "probability": 0.06
    }
  ]
}
```

---

## Results

A sample local training run achieved:

- **Accuracy:** `0.8750`

Generated output artifacts include:
- `models/apparel_classifier.joblib`
- `outputs/metrics.json`
- `outputs/confusion_matrix.png`
- `outputs/prediction_preview.csv`

---

## Example Use Cases

This type of workflow can support:

- apparel inventory categorization
- resale listing assistance
- product-image triage
- visual catalog cleanup
- lightweight image-based retail automation prototypes

---

## Limitations

This is a lightweight portfolio implementation, so it has a few intentional limitations:

- only 4 apparel classes
- balanced subset rather than full large-scale inventory data
- feature-based classifier instead of a CNN backbone
- single-label classification only
- local inference only in this version

This repository is meant to demonstrate the end-to-end workflow clearly and reproducibly, not to represent a full production-scale system.

---

## Future Improvements

Possible upgrades include:

- transfer learning with EfficientNet or ResNet
- multi-label classification
- larger class coverage
- background removal and object-focused cropping
- model explainability visualizations
- Docker deployment
- cloud inference deployment
- experiment tracking with MLflow
- batch inference endpoint

---

## Tech Stack

- Python
- FastAPI
- scikit-learn
- pandas
- NumPy
- Pillow
- matplotlib
- Hugging Face `datasets`
- joblib

---

## Notes

This project is a **portfolio version** built using a real public apparel dataset and a lightweight ML pipeline for reproducibility and easy local execution.

It is intended to demonstrate:
- practical dataset preparation
- computer vision feature engineering
- supervised classification
- API-based model serving

---

## Author

**Binduja Malempati**

If you use this project as part of a portfolio or interview discussion, focus on:
- the dataset preparation workflow
- the model pipeline design decisions
- why a lightweight baseline was chosen
- how the system could be extended toward a larger production architecture
