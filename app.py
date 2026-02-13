# app.py (Render UI -> proxy to HF Space API)
import os
import traceback
import requests
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# 👉 Base de ton Space HF (sans /docs)
DOMAIN_API_BASE = os.getenv("DOMAIN_API_BASE", "https://bineta123-domainexpert-api.hf.space").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))

# ----------------------------
# Always JSON for /api/*
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
# Upstream call helper
# ----------------------------
def call_domain_api(method: str, path: str, json_body=None, params=None):
    url = f"{DOMAIN_API_BASE}{path}"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "render-proxy/1.0"
    }

    try:
        r = requests.request(
            method=method,
            url=url,
            headers=headers,
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

    text = (r.text or "")
    if r.status_code >= 400:
        # Même si upstream renvoie HTML, on renvoie JSON côté Render
        return None, {
            "ok": False,
            "error": f"UpstreamHTTP{r.status_code}",
            "upstream": url,
            "status_code": r.status_code,
            "text_preview": text[:500]
        }, 502

    # parse JSON de manière safe
    if not text.strip():
        return None, {
            "ok": False,
            "error": "UpstreamEmptyBody",
            "upstream": url
        }, 502

    try:
        return r.json(), None, 200
    except Exception:
        return None, {
            "ok": False,
            "error": "UpstreamNotJSON",
            "upstream": url,
            "text_preview": text[:500]
        }, 502


# ----------------------------
# Pages (UI)
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    # Page simple (Playground)
    return render_template("playground.html")


# ----------------------------
# API routes expected by your frontend (Render)
# ----------------------------
@app.route("/api/ping", methods=["GET"])
def api_ping():
    data, err, code = call_domain_api("GET", "/ping")
    if err:
        return jsonify(err), code
    # data = {"ok": True, "msg": "pong"} depuis ton Space
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
    k = int(payload.get("k", 10))

    if not text:
        return jsonify({"ok": False, "error": "Missing 'text'"}), 400

    # Appelle ton Space HF
    data, err, code = call_domain_api("POST", "/predict", json_body={"text": text, "k": k})
    if err:
        return jsonify(err), code

    # Ton Space renvoie: {"ok": True, "top1": ..., "conf": ..., "topk": [...]}
    if isinstance(data, dict) and data.get("ok") is True:
        top1 = data.get("top1")
        topk = data.get("topk", [])
        conf = data.get("conf")

        # ✅ format compatible ancien + nouveau
        return jsonify({
            "ok": True,
            "final_category": top1,   # 👈 ton UI affiche "catégorie finale"
            "top1": top1,
            "top10": topk,
            "conf": conf
        }), 200

    return jsonify({"ok": False, "error": "Bad upstream response", "upstream": data}), 502


# (optionnel) route ping locale
@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
