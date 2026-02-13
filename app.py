# app.py (Render UI -> proxy to HF Space API) ✅ ROBUST

import os
import traceback
import requests

from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# Upstream HF Space base URL
DOMAIN_API_BASE = os.getenv("DOMAIN_API_BASE", "https://bineta123-domainexpert-api.hf.space").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))

# ----------------------------
# CORS (simple + enough)
# ----------------------------
@app.after_request
def add_cors_headers(resp):
    # allow calling from your own domain / browser
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp

@app.route("/api/<path:_>", methods=["OPTIONS"])
def preflight(_):
    # reply OK to preflight
    return ("", 204)

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
# Helper: call upstream HF API
# ----------------------------
def call_domain_api(method: str, path: str, json_body=None, params=None):
    url = f"{DOMAIN_API_BASE}{path}"

    try:
        r = requests.request(
            method=method,
            url=url,
            json=json_body,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
    except requests.RequestException as e:
        return None, {
            "ok": False,
            "error": f"UpstreamError: {type(e).__name__}: {str(e)}",
            "upstream": url
        }, 502

    # Try JSON; if not JSON, return preview
    try:
        data = r.json()
        return data, None, r.status_code
    except Exception:
        return None, {
            "ok": False,
            "error": "UpstreamNotJSON",
            "upstream": url,
            "status_code": r.status_code,
            "text_preview": (r.text or "")[:400]
        }, 502

# ----------------------------
# Pages (UI)
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    # If you have your template, keep it
    return render_template("domain_expert.html")

# ----------------------------
# API routes expected by your frontend
# ----------------------------
@app.route("/api/ping", methods=["GET"])
def api_ping():
    data, err, code = call_domain_api("GET", "/ping")
    if err:
        return jsonify(err), code
    # return same as upstream, but wrap
    return jsonify({"ok": True, "upstream": data}), code

@app.route("/api/health", methods=["GET"])
def api_health():
    data, err, code = call_domain_api("GET", "/health")
    if err:
        return jsonify(err), code
    return jsonify({"ok": True, "upstream": data}), code

@app.route("/api/health_model", methods=["GET"])
def api_health_model():
    data, err, code = call_domain_api("GET", "/health_model")
    if err:
        return jsonify(err), code
    return jsonify({"ok": True, "upstream": data}), code

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    k = payload.get("k", 10)

    if not text:
        return jsonify({"ok": False, "error": "Missing 'text'"}), 400

    # forward to HF
    upstream_data, err, status = call_domain_api(
        "POST",
        "/predict",
        json_body={"text": text, "k": int(k)}
    )

    if err:
        return jsonify(err), status

    # Upstream should be like: {"ok": True, "top1": ..., "conf": ..., "topk": [...]}
    if isinstance(upstream_data, dict) and upstream_data.get("ok") is True:
        return jsonify({
            "ok": True,
            "top1": upstream_data.get("top1"),
            "conf": upstream_data.get("conf"),
            "top10": upstream_data.get("topk", []),
        }), 200

    # If upstream returned structured error
    return jsonify({
        "ok": False,
        "error": "Bad upstream response",
        "upstream_status": status,
        "upstream": upstream_data
    }), 502


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
