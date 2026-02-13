# app.py (Render UI -> proxy to HF Space API)

from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException
import os, traceback
import requests

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
# Helpers
# ----------------------------
def call_domain_api(method: str, path: str, json_body=None, params=None):
    url = f"{DOMAIN_API_BASE}{path}"
    try:
        r = requests.request(
            method=method,
            url=url,
            json=json_body,
            params=params,
            timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as e:
        return None, {
            "ok": False,
            "error": f"UpstreamError: {type(e).__name__}: {str(e)}",
            "upstream": url
        }, 502

    # Toujours essayer JSON, sinon renvoyer texte
    try:
        data = r.json()
        return data, None, r.status_code
    except Exception:
        return None, {
            "ok": False,
            "error": "UpstreamNotJSON",
            "upstream": url,
            "status_code": r.status_code,
            "text_preview": (r.text or "")[:300]
        }, 502


# ----------------------------
# Pages (UI)
# ----------------------------
@app.route("/")
def home():
    # si tu as déjà un template, garde-le
    # sinon tu peux mettre une simple page
    return render_template("domain_expert.html")


# ----------------------------
# API routes expected by your frontend
# ----------------------------
@app.route("/api/ping")
def api_ping():
    data, err, code = call_domain_api("GET", "/ping")
    if err:
        return jsonify(err), code
    return jsonify({"ok": True, "upstream": data}), 200


@app.route("/api/health")
def api_health():
    data, err, code = call_domain_api("GET", "/health")
    if err:
        return jsonify(err), code
    return jsonify({"ok": True, "upstream": data}), 200


@app.route("/api/health_model")
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

    # Normalize output to what your frontend expects
    # HF returns: {"ok": True, "top1": ..., "conf": ..., "topk": [...]}
    if isinstance(data, dict) and data.get("ok") is True:
        return jsonify({
            "ok": True,
            "top1": data.get("top1"),
            "conf": data.get("conf"),
            "top10": data.get("topk", [])  # keep same naming as your old UI
        }), 200

    return jsonify({"ok": False, "error": "Bad upstream response", "upstream": data}), 502


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
