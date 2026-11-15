export const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function predict(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API}/predict`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || "Prediction failed");
  return res.json();
}

export async function sendDecision(scanId, decision, reviewer, note) {
  const res = await fetch(`${API}/decisions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scan_id: scanId, decision, reviewer, note }),
  });
  if (!res.ok) throw new Error("Could not save decision");
  return res.json();
}

export async function fetchAudit() {
  const res = await fetch(`${API}/audit`);
  return res.json();
}
