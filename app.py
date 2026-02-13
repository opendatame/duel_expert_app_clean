# app.py (Render UI -> proxy to HF Space API)

import os
import traceback
import requests
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# URL de ton HF Space (API FastAPI)
DOMAIN_API_BASE = os.getenv("DOMAIN_API_BASE", "https://bineta123-domainexpert-api.hf.space").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))  # plus long (cold start HF)

session = requests.Session()


# ----------------------------
# ALWAYS JSON FOR /api/*
# ----------------------------
@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    if request.path.startswith("/api/"):
        return jsonify({
            "ok": False,
            "error": f"{type(e).__name__}: {e.description}",
            "path": request.path
        }), e.code
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
# Proxy helper
# ----------------------------
def call_domain_api(method: str, path: str, json_body=None):
    url = f"{DOMAIN_API_BASE}{path}"

    try:
        r = session.request(
            method=method,
            url=url,
            json=json_body,
            timeout=REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
    except requests.RequestException as e:
        return None, {
            "ok": False,
            "error": f"UpstreamError: {type(e).__name__}: {str(e)}",
            "upstream": url
        }, 502

    txt = r.text or ""

    # Si upstream renvoie HTML/vides, on renvoie quand même du JSON clair
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

    # Si upstream renvoie une erreur, on la propage en JSON
    if r.status_code >= 400:
        return None, {
            "ok": False,
            "error": "UpstreamHTTPError",
            "upstream": url,
            "status_code": r.status_code,
            "upstream_json": data
        }, 502

    return data, None, r.status_code


# ----------------------------
# Pages (UI)
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    # ton template UI
    return render_template("domain_expert.html")


# ----------------------------
# API routes used by your frontend
# ----------------------------
@app.route("/api/ping", methods=["GET"])
def api_ping():
    data, err, code = call_domain_api("GET", "/ping")
    if err:
        return jsonify(err), code
    return jsonify({"ok": True, "upstream": data, "base": DOMAIN_API_BASE}), 200


@app.route("/api/health", methods=["GET"])
def api_health():
    data, err, code = call_domain_api("GET", "/health")
    if err:
        return jsonify(err), code
    return jsonify({"ok": True, "upstream": data}), 200


@app.route("/api/health_model", methods=["GET"])
def api_health_model():
    data, err, code = call_domain_api("GET", "/health_model")
    if err:
        return jsonify(err), code
    return jsonify({"ok": True, "upstream": data}), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    k = int(payload.get("k", 10))

    if not text:
        return jsonify({"ok": False, "error": "Missing 'text'"}), 400

    data, err, code = call_domain_api("POST", "/predict", {"text": text, "k": k})
    if err:
        return jsonify(err), code

    # HF FastAPI renvoie: {"ok": True, "top1": ..., "conf": ..., "topk": [...]}
    return jsonify({
        "ok": True,
        "top1": data.get("top1"),
        "conf": data.get("conf"),
        "top10": data.get("topk", [])
    }), 200


# petit ping Render (utile pour tester vite)
@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200
