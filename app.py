# app.py (Render UI -> proxy to HF Space API)

import os
import traceback

import requests
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

DOMAIN_API_BASE = os.getenv("DOMAIN_API_BASE", "https://bineta123-domainexpert-api.hf.space").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))


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
# UPSTREAM CALL
# ----------------------------
def call_domain_api(method: str, path: str, json_body=None):
    url = f"{DOMAIN_API_BASE}{path}"

    try:
        r = requests.request(
            method=method,
            url=url,
            json=json_body,
            timeout=REQUEST_TIMEOUT,
            headers={
                "Accept": "application/json",
                "User-Agent": "Render-Proxy/1.0"
            },
        )
    except requests.RequestException as e:
        return None, {
            "ok": False,
            "error": f"UpstreamError: {type(e).__name__}: {str(e)}",
            "upstream": url
        }, 502

    txt = r.text or ""
    # Upstream status code is useful
    status = r.status_code

    # Try JSON if possible
    if txt.strip():
        try:
            data = r.json()
            return data, None, status
        except Exception:
            # Upstream returned HTML/text -> still return JSON to frontend
            return None, {
                "ok": False,
                "error": "UpstreamNotJSON",
                "upstream": url,
                "status_code": status,
                "content_type": r.headers.get("content-type"),
                "text_preview": txt[:400],
            }, 502
    else:
        # Empty body -> still return JSON
        return None, {
            "ok": False,
            "error": "UpstreamEmptyBody",
            "upstream": url,
            "status_code": status,
        }, 502


# ----------------------------
# PAGES
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    # Ton template UI
    return render_template("domain_expert.html")


# ----------------------------
# API (frontend expects these)
# ----------------------------
@app.route("/api/ping", methods=["GET"])
def api_ping():
    data, err, code = call_domain_api("GET", "/ping")
    if err:
        return jsonify(err), code
    return jsonify({"ok": True, "upstream": data}), 200


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
    text = (payload.get("text") or "").strip()
    k = payload.get("k", 10)

    if not text:
        return jsonify({"ok": False, "error": "Missing 'text'"}), 400

    data, err, code = call_domain_api("POST", "/predict", json_body={"text": text, "k": int(k)})
    if err:
        return jsonify(err), code

    # HF Space returns: {"ok": True, "top1": ..., "conf": ..., "topk": [...]}
    if isinstance(data, dict) and data.get("ok") is True:
        return jsonify({
            "ok": True,
            "top1": data.get("top1"),
            "conf": data.get("conf"),
            "top10": data.get("topk", []),
        }), 200

    return jsonify({"ok": False, "error": "Bad upstream response", "upstream": data}), 502


# legacy ping (optionnel)
@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200
