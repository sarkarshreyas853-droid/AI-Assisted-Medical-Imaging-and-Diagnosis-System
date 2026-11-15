import React, { useEffect, useState } from "react";
import { predict, sendDecision, fetchAudit } from "./api.js";

const DISCLAIMER =
  "For education only. Not for clinical use. Outputs must not inform diagnosis or treatment.";

export default function App() {
  const [scan, setScan] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [decided, setDecided] = useState(null);
  const [audit, setAudit] = useState([]);

  useEffect(() => {
    fetchAudit().then(setAudit).catch(() => {});
  }, []);

  async function onUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError("");
    setDecided(null);
    try {
      setScan(await predict(file));
    } catch (err) {
      setError(err.message);
      setScan(null);
    } finally {
      setBusy(false);
    }
  }

  async function decide(kind) {
    if (!scan) return;
    try {
      await sendDecision(scan.scan_id, kind, reviewer || "anonymous");
      setDecided(kind);
      setAudit(await fetchAudit());
    } catch (err) {
      setError(err.message);
    }
  }

  const src = (b64) => `data:image/png;base64,${b64}`;

  return (
    <main className="wrap">
      <header>
        <h1>AI-Assisted Imaging Review</h1>
        <p className="disclaimer">{DISCLAIMER}</p>
      </header>

      <section className="card controls">
        <label className="upload">
          <input type="file" accept=".png,.jpg,.jpeg,.dcm" onChange={onUpload} />
          <span>Choose a chest X-ray (PNG, JPG or DICOM)</span>
        </label>
        <label className="reviewer">
          Reviewer name
          <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="e.g. Dr. Rao" />
        </label>
      </section>

      {busy && <p className="status">Analysing scan...</p>}
      {error && <p className="error">{error}</p>}

      {scan && (
        <section className="card result">
          <div className="images">
            <div className="frame">
              <img src={src(scan.original_png_b64)} alt="Scan" />
              {showHeatmap && <img className="overlay" src={src(scan.heatmap_png_b64)} alt="Grad-CAM heatmap" />}
            </div>
            <div className="details">
              <p className="label">
                Prediction: <strong>{scan.label}</strong>
              </p>
              <p>Confidence: <strong>{(scan.confidence * 100).toFixed(1)}%</strong></p>
              {!scan.classifier_trained && (
                <p className="warn">Classifier not trained yet. This output is random and for pipeline testing only.</p>
              )}
              <label className="toggle">
                <input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} />
                Show heatmap (regions that influenced the prediction)
              </label>
              <div className="actions">
                <button onClick={() => decide("approve")} disabled={!!decided}>Approve</button>
                <button className="secondary" onClick={() => decide("reject")} disabled={!!decided}>Reject</button>
              </div>
              {decided && <p className="status">Recorded: {decided}. Logged for audit.</p>}
            </div>
          </div>
        </section>
      )}

      <section className="card">
        <h2>Audit log</h2>
        {audit.length === 0 && <p>No events yet.</p>}
        <ul className="audit">
          {audit.slice(-20).reverse().map((e, i) => (
            <li key={i}>
              <code>{e.timestamp?.slice(0, 19).replace("T", " ")}</code> {e.event}
              {e.decision ? ` · ${e.decision} by ${e.reviewer}` : ` · ${e.label} (${(e.confidence * 100).toFixed(1)}%)`}
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
