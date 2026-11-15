# AI-Assisted Medical Imaging and Diagnosis System (education demo)

> **For education only. Not for clinical use.** Outputs must not inform diagnosis or treatment.

A radiologist-in-the-loop prototype. A scan is classified (normal vs. pneumonia) with a fine-tuned DenseNet121, explained with Grad-CAM heatmaps, and can be segmented with a 3D U-Net (MONAI, spleen task). A reviewer approves or rejects each result, and every decision is written to an audit log.

## Architecture

```
React viewer ──► FastAPI ──► DenseNet121 + Grad-CAM   (chest X-ray classification)
                    │
                    ├──► MONAI 3D U-Net                (spleen segmentation, Dice)
                    └──► audit log (JSON lines)        (predictions + reviewer decisions)

Training: training/*.py  ·  Data versioning: DVC (dvc.yaml)  ·  Packaging: Docker
```

## Data

- Classification: NIH ChestX-ray14 subset or a Kaggle pneumonia dataset, placed in `data/raw/chest/{normal,pneumonia}/`.
- Segmentation: Medical Segmentation Decathlon Task09_Spleen, downloaded by MONAI into `data/raw/msd/`.

Data is not stored in git. Track it with DVC (`dvc add data/raw/chest`) after you download it.

## Run it

Train (GPU recommended, e.g. Google Colab):

```bash
python training/train_classifier.py --data data/raw/chest --epochs 5
python training/train_segmenter.py --root data/raw/msd --epochs 50
```

Serve:

```bash
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
cd frontend && npm install && npm run dev     # open http://localhost:5173
```

Or with Docker: `docker compose up --build`.

Tests: `pytest backend/tests`.

## Results

Fill these in after training:

| Model | Metric | Value |
|---|---|---|
| DenseNet121 (validation, epoch 5) | Accuracy | 0.978 |
| DenseNet121 (validation, epoch 5) | AUC | 0.998 |
| 3D U-Net spleen (validation) | Dice | Not yet trained (pipeline smoke-tested only) |

Note: the train and validation split is random, so the same patient may appear in both. These numbers may overstate performance on new patients. This is an educational prototype and not a clinical tool.

## Limitations

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for dataset bias, label noise, the limits of Grad-CAM, and the fact that this is not a medical device.

## Status

- Implemented: preprocessing (PNG, JPEG, DICOM), classifier, Grad-CAM, segmentation, API, audit log, React viewer, tests, Docker, DVC pipeline.
- Not yet: Orthanc DICOM server integration (planned next step).
