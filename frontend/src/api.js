export const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const WAKE_RETRY_DELAYS_MS = [4000, 10000, 20000];
const REQUEST_TIMEOUT_MS = 40000;

// The backend may be asleep on a free host and take a while to wake up, so a
// network failure is retried a few times before giving up with a clear message.
export async function predict(file) {
  let lastError;
  for (let attempt = 0; attempt <= WAKE_RETRY_DELAYS_MS.length; attempt++) {
    if (attempt > 0) {
      await new Promise((resolve) => setTimeout(resolve, WAKE_RETRY_DELAYS_MS[attempt - 1]));
    }
    const form = new FormData();
    form.append("file", file);
    let res;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      res = await fetch(`${API}/predict`, { method: "POST", body: form, signal: controller.signal });
    } catch (err) {
      lastError = err;
      continue; // network failure or stalled request: try again
    } finally {
      clearTimeout(timer);
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || "Prediction failed");
    }
    return res.json();
  }
  throw new Error(
    "Could not reach the analysis server. It may be waking up after inactivity. Wait a minute and try again."
  );
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
