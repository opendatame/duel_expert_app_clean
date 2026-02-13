# ================================
# app.py (Render) - UI + Proxy to Hugging Face API (NO torch/transformers)
# ================================

from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException
import os, time, traceback
import requests

app = Flask(__name__)

# ----------------------------
# CONFIG
# ----------------------------
HF_API_BASE = os.getenv("HF_API_BASE", "https://bineta123-domainexpert-api.hf.space")
HF_TIMEOUT  = float(os.getenv("HF_TIMEOUT", "40"))

# Optional: if your HF Space is private, put a token in Render env: HF_SPACE_TOKEN
HF_SPACE_TOKEN = os.getenv("HF_SPACE_TOKEN")  # optional

def _headers():
    h = {"Content-Type": "application/json"}
    if HF_SPACE_TOKEN:
        h["Authorization"] = f"Bearer {HF_SPACE_TOKEN}"
    return h

# ----------------------------
# ALWAYS JSON FOR /api/*
# ----------------------------
@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e.description}", "path": request.path}), e.code
    return e

@app.errorhandler(Exception)
def handle_any_exception(e):
    if request.path.startswith("/api/"):
        return jsonify({
            "ok": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "path": request.path,
            "trace": traceback.format_exc().splitlines()[-12:]
        }), 500
    raise e

# ----------------------------
# PROXY HELPERS
# ----------------------------
def hf_get(path: str):
    url = HF_API_BASE.rstrip("/") + path
    r = requests.get(url, headers=_headers(), timeout=HF_TIMEOUT)
    return r.status_code, r.json() if "application/json" in r.headers.get("content-type", "") else {"raw": r.text}

def hf_post(path: str, payload: dict):
    url = HF_API_BASE.rstrip("/") + path
    r = requests.post(url, headers=_headers(), json=payload, timeout=HF_TIMEOUT)
    return r.status_code, r.json() if "application/json" in r.headers.get("content-type", "") else {"raw": r.text}

# ----------------------------
# PAGES (minimal)
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    # Si tu as déjà des templates, remets ton render_template ici.
    # Sinon, une page simple:
    return (
        "<h2>Render UI (proxy) ✅</h2>"
        "<p>HF API: <code>{}</code></p>"
        "<p>Try: <code>/api/ping</code> or POST <code>/api/predict</code></p>".format(HF_API_BASE)
    )

# ----------------------------
# API (routes expected by your frontend)
# ----------------------------
@app.route("/api/ping", methods=["GET"])
def api_ping():
    status, data = hf_get("/ping")
    return jsonify({"ok": status == 200, "hf_status": status, "hf": data, "hf_base": HF_API_BASE}), (200 if status == 200 else 502)

@app.route("/api/health_model", methods=["GET"])
def api_health_model():
    status, data = hf_get("/health_model")
    return jsonify({"ok": status == 200, "hf_status": status, "hf": data}), (200 if status == 200 else 502)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    k = body.get("k", 10)

    if not text:
        return jsonify({"ok": False, "error": "Missing 'text'"}), 400

    t0 = time.time()
    status, data = hf_post("/predict", {"text": text, "k": int(k)})

    # normalize response
    out = {
        "ok": status == 200 and bool(data.get("ok", True)),
        "latency_sec": round(time.time() - t0, 2),
        "hf_status": status,
        "hf_response": data
    }
    return jsonify(out), (200 if out["ok"] else 502)

# Compatibility (if something calls without /api)
@app.route("/ping", methods=["GET"])
def ping():
    return api_ping()

@app.route("/health_model", methods=["GET"])
def health_model():
    return api_health_model()

@app.route("/predict", methods=["POST"])
def predict():
    return api_predict()

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
