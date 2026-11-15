import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import os
os.environ.setdefault("AUDIT_LOG", "logs/test_audit.jsonl")

from backend.app.main import app  # noqa: E402

client = TestClient(app)


def _png_bytes(size=(256, 256)) -> bytes:
    arr = (np.random.rand(*size) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health_has_disclaimer():
    r = client.get("/health")
    assert r.status_code == 200
    assert "Not for clinical use" in r.json()["disclaimer"]


def test_predict_returns_label_confidence_and_heatmap():
    r = client.post("/predict", files={"file": ("scan.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("normal", "pneumonia")
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["heatmap_png_b64"]) > 100
    assert body["scan_id"]


def test_predict_rejects_empty_file():
    r = client.post("/predict", files={"file": ("scan.png", b"", "image/png")})
    assert r.status_code == 400


def test_decision_is_logged():
    r = client.post("/decisions", json={"scan_id": "abc", "decision": "approve", "reviewer": "dr-test"})
    assert r.status_code == 200
    log = client.get("/audit").json()
    assert any(e.get("scan_id") == "abc" and e.get("decision") == "approve" for e in log)


def test_invalid_decision_rejected():
    r = client.post("/decisions", json={"scan_id": "abc", "decision": "maybe"})
    assert r.status_code == 422
