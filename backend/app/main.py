"""FastAPI service: classify a scan, explain it with Grad-CAM, record decisions.

FOR EDUCATION ONLY. NOT FOR CLINICAL USE.
"""
from __future__ import annotations

import base64
import io
import uuid
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import audit
from .classifier import Classifier
from .imaging import load_image, to_tensor

DISCLAIMER = "For education only. Not for clinical use. Outputs must not inform diagnosis or treatment."

app = FastAPI(title="AI-Assisted Imaging (education demo)", description=DISCLAIMER)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_classifier() -> Classifier:
    return Classifier()


class Decision(BaseModel):
    scan_id: str
    decision: str  # "approve" or "reject"
    reviewer: str = "anonymous"
    note: str = ""


@app.get("/health")
def health():
    clf = get_classifier()
    return {"status": "ok", "classifier_trained": clf.trained, "disclaimer": DISCLAIMER}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        image = load_image(data, file.filename or "upload.png")
    except Exception as exc:  # pragma: no cover - depends on uploaded bytes
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    clf = get_classifier()
    x = to_tensor(image)
    label, confidence, idx = clf.predict(x)
    heatmap_b64 = clf.gradcam(x, idx, image)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    scan_id = str(uuid.uuid4())
    audit.record({"event": "prediction", "scan_id": scan_id, "filename": file.filename,
                  "label": label, "confidence": round(confidence, 4),
                  "classifier_trained": clf.trained})
    return {
        "scan_id": scan_id,
        "label": label,
        "confidence": round(confidence, 4),
        "classifier_trained": clf.trained,
        "original_png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "heatmap_png_b64": heatmap_b64,
        "disclaimer": DISCLAIMER,
    }


@app.post("/decisions")
def decide(body: Decision):
    if body.decision not in ("approve", "reject"):
        raise HTTPException(status_code=422, detail="decision must be 'approve' or 'reject'")
    return audit.record({"event": "decision", **body.model_dump()})


@app.get("/audit")
def get_audit():
    return audit.read_all()
