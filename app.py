# ================================
# app.py – Render UI (NO model load)
# Calls HF Space API: Bineta123/Domainexpert-api
# ================================

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.exceptions import HTTPException
import os, time, traceback
import requests
import pandas as pd

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR   = os.path.join(BASE_DIR, "data")

BACKGROUND_IMAGE_FILE = os.getenv("BACKGROUND_IMAGE_FILE", "imageeco.jpg")

# ✅ Your HF Space API base URL
API_BASE = os.getenv("DOMAINEXPERT_API_BASE", "https://bineta123-domainexpert-api.hf.space").rstrip("/")

# Optional CSVs (only for UI/metrics/examples if you want)
DOMAIN_CSV = os.getenv("DOMAIN_CSV", os.path.join(DATA_DIR, "domain_expert_phase2_top10_predictions_full.csv"))
DUEL_CSV   = os.getenv("DUEL_CSV",   os.path.join(DATA_DIR, "duel_expert_mistral_GENERAL_EXPERT_top10_corrected.csv"))

TIMEOUT_SEC = float(os.getenv("API_TIMEOUT_SEC", "30"))


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
# Helpers
# ----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df

def safe_read_csv(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
        return normalize_columns(df)
    except Exception:
        try:
            df = pd.read_csv(path, sep=";", on_bad_lines="skip")
            return normalize_columns(df)
        except Exception:
            try:
                df = pd.read_csv(path, sep=",", on_bad_lines="skip")
                return normalize_columns(df)
            except Exception:
                return None


def call_hf_predict(text: str, k: int = 10):
    """
    Calls HF Space FastAPI:
    POST {API_BASE}/predict
    body: {"text": "...", "k": 10}
    returns json
    """
    url = f"{API_BASE}/predict"
    payload = {"text": text, "k": int(k)}
    r = requests.post(url, json=payload, timeout=TIMEOUT_SEC)
    # If HF returns non-200, raise a readable error
    if r.status_code != 200:
        raise RuntimeError(f"HF API error {r.status_code}: {r.text[:300]}")
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"HF API returned ok=false: {data}")
    return data  # {"ok": True, "top1": ..., "conf": ..., "topk": [...]}


# ----------------------------
# Routes (API for frontend)
# ----------------------------
@app.route("/api/ping")
def api_ping():
    # ping Render + ping HF
    hf_ok = False
    hf_msg = ""
    try:
        rr = requests.get(f"{API_BASE}/ping", timeout=10)
        hf_ok = rr.status_code == 200
        hf_msg = rr.text[:120]
    except Exception as e:
        hf_msg = f"{type(e).__name__}: {str(e)}"

    return jsonify({
        "ok": True,
        "render": "pong",
        "hf_base": API_BASE,
        "hf_reachable": hf_ok,
        "hf_msg": hf_msg
    }), 200


@app.route("/api/health")
def api_health():
    # just proxy HF health + small info
    out = {"ok": True, "hf_base": API_BASE}
    try:
        rr = requests.get(f"{API_BASE}/health", timeout=10)
        out["hf_health_status"] = rr.status_code
        if rr.headers.get("content-type", "").startswith("application/json"):
            out["hf_health"] = rr.json()
        else:
            out["hf_health"] = rr.text[:300]
    except Exception as e:
        out["hf_health_error"] = f"{type(e).__name__}: {str(e)}"
    return jsonify(out), 200


@app.route("/api/health_model")
def api_health_model():
    # proxy HF health_model
    rr = requests.get(f"{API_BASE}/health_model", timeout=TIMEOUT_SEC)
    return (rr.text, rr.status_code, {"Content-Type": rr.headers.get("content-type", "application/json")})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    k = int(data.get("k", 10))

    if not text:
        return jsonify({"ok": False, "error": "Missing 'text'"}), 400

    t0 = time.time()
    pred = call_hf_predict(text, k=k)

    return jsonify({
        "ok": True,
        "input_text": text,
        "final_category": pred["top1"],
        "conf": float(pred.get("conf", 0.0)),
        "topk": pred.get("topk", []),
        "latency_sec": round(time.time() - t0, 2),
        "provider": "hf-space"
    }), 200


# ----------------------------
# Pages (UI)
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_results = []
    llm_msg = ""
    start_time = time.time()
    background_image_url = url_for("static", filename=BACKGROUND_IMAGE_FILE)

    products = []
    if request.method == "POST":
        uploaded_file = request.files.get("csv_file")
        new_product = (request.form.get("product_text") or "").strip()

        if uploaded_file:
            df_new = safe_read_csv(uploaded_file)  # not valid: file object. We'll handle below
            # read directly from uploaded file:
            try:
                df_new = pd.read_csv(uploaded_file, sep=None, engine="python", on_bad_lines="skip")
                df_new = normalize_columns(df_new)
            except Exception:
                uploaded_file.stream.seek(0)
                df_new = pd.read_csv(uploaded_file, sep=";", on_bad_lines="skip")
                df_new = normalize_columns(df_new)

            if "description" in df_new.columns and "text" not in df_new.columns:
                df_new.rename(columns={"description": "text"}, inplace=True)

            if "text" not in df_new.columns:
                llm_msg = "CSV doit contenir une colonne 'text' (ou 'description')."
            else:
                products = df_new["text"].astype(str).fillna("").tolist()

        elif new_product:
            products = [new_product]

    if products:
        try:
            for prod in products:
                prod = str(prod).strip()
                if not prod:
                    continue
                pred = call_hf_predict(prod, k=10)
                prediction_results.append({
                    "text": prod,
                    "top1": pred["top1"],
                    "final": pred["top1"],
                    "top10": pred.get("topk", [])[:10],
                    "conf": round(float(pred.get("conf", 0.0)), 4)
                })
            llm_msg = "✅ Prédiction OK (HF API)"
        except Exception as e:
            llm_msg = f"❌ Erreur HF API: {type(e).__name__}: {str(e)}"

    elapsed_time = round(time.time() - start_time, 2)

    # If you want example products from DUEL_CSV (optional)
    df_duel = safe_read_csv(DUEL_CSV)
    example_products = df_duel.head(10).to_dict(orient="records") if df_duel is not None else []

    return render_template(
        "domain_expert.html",
        background_image_url=background_image_url,
        prediction_results=prediction_results,
        llm_msg=llm_msg,
        elapsed_time=elapsed_time,
        products=example_products,
        # keep these for template compatibility (can be None)
        domain_metrics=None,
        duel_metrics=None,
        plot_html=None
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
