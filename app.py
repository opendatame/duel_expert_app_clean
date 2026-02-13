# app.py (Render UI -> proxy to HF Space API)

from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException
import os, traceback, requests

app = Flask(__name__)

DOMAIN_API_BASE = os.getenv("DOMAIN_API_BASE", "https://bineta123-domainexpert-api.hf.space").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))

# ---- always JSON for /api/*
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

def call_domain_api(method: str, path: str, json_body=None):
    url = f"{DOMAIN_API_BASE}{path}"
    try:
        r = requests.request(method, url, json=json_body, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        return None, {"ok": False, "error": f"UpstreamError: {type(e).__name__}: {str(e)}", "upstream": url}, 502

    # read text first to avoid json() crash
    txt = r.text or ""
    try:
        data = r.json() if txt.strip() else {}
    except Exception:
        return None, {
            "ok": False,
            "error": "UpstreamNotJSON",
            "upstream": url,
            "status_code": r.status_code,
            "text_preview": txt[:300]
        }, 502

    return data, None, r.status_code

# ---- pages
@app.route("/")
def home():
    # si tu as ton template:
    return render_template("domain_expert.html")

# ---- API expected by your frontend
@app.route("/api/ping", methods=["GET"])
def api_ping():
    data, err, code = call_domain_api("GET", "/ping")
    return (jsonify(err), code) if err else (jsonify(data), code)

@app.route("/api/health", methods=["GET"])
def api_health():
    data, err, code = call_domain_api("GET", "/health")
    return (jsonify(err), code) if err else (jsonify(data), code)

@app.route("/api/health_model", methods=["GET"])
def api_health_model():
    data, err, code = call_domain_api("GET", "/health_model")
    return (jsonify(err), code) if err else (jsonify(data), code)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    k = int(payload.get("k", 10))

    if not text:
        return jsonify({"ok": False, "error": "Missing 'text'"}), 400

    data, err, code = call_domain_api("POST", "/predict", {"text": text, "k": k})
    if err:
        return jsonify(err), code

    # normalize for your old UI
    if isinstance(data, dict) and data.get("ok") is True:
        return jsonify({
            "ok": True,
            "top1": data.get("top1"),
            "conf": data.get("conf"),
            "top10": data.get("topk", [])
        }), 200

    return jsonify({"ok": False, "error": "Bad upstream response", "upstream": data}), 502

# legacy ping (optional)
@app.route("/ping")
def ping():
    return "pong", 200
