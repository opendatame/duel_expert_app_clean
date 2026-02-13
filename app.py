# ================================
# app.py – Duel Expert Flask App (Render-ready)
# - CSV in /data (GitHub)
# - HF .pth download (public or private)
# - Adds /predict JSON endpoint to avoid "Unexpected end of JSON input"
# - Adds /health_model debug endpoint
# - Adds HF/Transformers cache dirs to reduce repeated downloads
# ================================

from flask import Flask, render_template, request, jsonify, url_for
import os, time, ast, gc, shutil
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import plotly.graph_objs as go
import plotly.io as pio

from huggingface_hub import hf_hub_download
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel


# ----------------------------
# CONFIG & PATHS (portable)
# ----------------------------
app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Cache HF/Transformers locally (helps on Render)
HF_CACHE_DIR = os.path.join(BASE_DIR, ".hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ✅ Render: static background image
BACKGROUND_IMAGE_FILE = os.getenv("BACKGROUND_IMAGE_FILE", "imageeco.jpg")

# ✅ CSV paths (you have /data in repo)
DOMAIN_CSV = os.getenv("DOMAIN_CSV", os.path.join(DATA_DIR, "domain_expert_phase2_top10_predictions_full.csv"))
DUEL_CSV   = os.getenv("DUEL_CSV",   os.path.join(DATA_DIR, "duel_expert_mistral_GENERAL_EXPERT_top10_corrected.csv"))
GLOBAL_CSV = os.getenv("GLOBAL_CSV", os.path.join(DATA_DIR, "produits_nettoyes.csv"))

# ✅ Local where we store downloaded .pth
PHASE2_CKPT = os.getenv("PHASE2_CKPT", os.path.join(MODEL_DIR, "domain_expert_flat.pth"))

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = int(os.getenv("MAX_LEN", "160"))

# ----------------------------
# HF MODEL DOWNLOAD (.pth)
# ----------------------------
HF_REPO_ID  = os.getenv("HF_REPO_ID", "Bineta123/domain-expert-xlmr")
HF_FILENAME = os.getenv("HF_FILENAME", "domain_expert_flat.pth")
HF_TOKEN    = os.getenv("HF_TOKEN")  # if private, set as Render Secret. if public, can be None.


def ensure_model_file(local_path: str):
    """Download .pth from Hugging Face if missing/too small."""
    if os.path.exists(local_path):
        try:
            if os.path.getsize(local_path) > 10_000_000:
                print("[MODEL] already present:", local_path, "size=", os.path.getsize(local_path))
                return
        except Exception:
            pass

    print("[MODEL] downloading .pth from Hugging Face...")
    # token can be None for public repos
    downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, token=HF_TOKEN)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copyfile(downloaded, local_path)

    try:
        print("[MODEL] ✅ downloaded to:", local_path, "size=", os.path.getsize(local_path))
    except Exception:
        print("[MODEL] ✅ downloaded to:", local_path)


# ----------------------------
# SAFE CSV READER + NORMALIZE COLS
# ----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def safe_read_csv(path: str):
    if not path or not os.path.exists(path):
        print("[CSV] missing:", path)
        return None

    try:
        if os.path.getsize(path) < 10:
            print("[CSV] empty:", path)
            return None
    except Exception:
        pass

    # try sniff
    try:
        df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
        df = normalize_columns(df)
        if df.shape[1] > 0:
            return df
    except Exception as e:
        print("[CSV] sniff failed:", type(e).__name__, str(e))

    for sep in [";", ","]:
        try:
            df = pd.read_csv(path, sep=sep, on_bad_lines="skip")
            df = normalize_columns(df)
            if df.shape[1] > 0:
                return df
        except Exception as e:
            print("[CSV] read failed sep=", sep, type(e).__name__, str(e))

    return None


# ----------------------------
# DOMAIN EXPERT MODEL
# ----------------------------
class DomainExpert(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        # ⚠️ large is heavy on Render Free. Keep if your checkpoint is large-based.
        # If you switch to base, your .pth must match base.
        self.xlm = XLMRobertaModel.from_pretrained("xlm-roberta-large")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.xlm.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.xlm(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


# ----------------------------
# LAZY GLOBALS (avoid heavy boot)
# ----------------------------
tokenizer = None
le = None
NUM_CLASSES = None
model = None

df_domain = None
df_duel = None
domain_metrics = None
duel_metrics = None
plot_metrics_html = None


def load_tokenizer_if_needed():
    global tokenizer
    if tokenizer is None:
        print("[LOAD] tokenizer xlm-roberta-large...")
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")


def load_label_encoder_if_needed():
    global le, NUM_CLASSES
    if le is not None and NUM_CLASSES is not None:
        return

    df_global = safe_read_csv(GLOBAL_CSV)
    if df_global is None:
        raise RuntimeError(f"GLOBAL_CSV unreadable: {GLOBAL_CSV}")

    if "taxonomy_path" not in df_global.columns:
        raise ValueError("GLOBAL_CSV must contain column 'taxonomy_path'")

    df_global["taxonomy_path"] = df_global["taxonomy_path"].astype(str)

    le_local = LabelEncoder()
    le_local.fit(df_global["taxonomy_path"].tolist())
    le = le_local
    NUM_CLASSES = len(le.classes_)
    print("[INIT] NUM_CLASSES:", NUM_CLASSES)


def extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt


def load_domain_expert_if_needed():
    global model
    if model is not None:
        return

    load_label_encoder_if_needed()
    load_tokenizer_if_needed()
    ensure_model_file(PHASE2_CKPT)

    print("[LOAD] creating model...")
    model_local = DomainExpert(NUM_CLASSES).to(DEVICE)

    print("[LOAD] loading weights from:", PHASE2_CKPT)
    ckpt = torch.load(PHASE2_CKPT, map_location=DEVICE)
    sd = extract_state_dict(ckpt)
    if isinstance(sd, dict):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # strict=False to tolerate small key mismatches
    missing, unexpected = model_local.load_state_dict(sd, strict=False)
    print("[LOAD] missing keys:", len(missing), "unexpected keys:", len(unexpected))

    model_local.eval()

    del ckpt, sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = model_local
    print("[LOAD] ✅ model ready on", DEVICE)


@torch.no_grad()
def predict_topk(text: str, k: int = 10):
    load_domain_expert_if_needed()

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = F.softmax(logits, dim=-1).squeeze(0)

    kk = max(1, min(int(k), int(probs.shape[0])))
    top_probs, top_idx = torch.topk(probs, k=kk)

    labels = le.inverse_transform(top_idx.detach().cpu().numpy()).tolist()
    probs_list = top_probs.detach().cpu().numpy().tolist()
    return labels, probs_list


# ----------------------------
# OPTIONAL ASSETS + METRICS (ROBUST)
# ----------------------------
def pick_true_col(df: pd.DataFrame):
    candidates = ["true_label", "label", "y_true", "gold", "ground_truth"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_optional_assets_if_needed():
    global df_domain, df_duel, domain_metrics, duel_metrics, plot_metrics_html

    if df_domain is not None or df_duel is not None:
        return

    df_domain = safe_read_csv(DOMAIN_CSV)
    df_duel   = safe_read_csv(DUEL_CSV)

    for df in [df_domain, df_duel]:
        if df is None:
            continue
        if "description" in df.columns and "text" not in df.columns:
            df.rename(columns={"description": "text"}, inplace=True)

        if "top10_preds" in df.columns:
            df["top10_preds"] = df["top10_preds"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

    def compute_metrics(df, pred_col):
        true_col = pick_true_col(df)
        if true_col is None or pred_col is None:
            return None
        if pred_col not in df.columns:
            return None

        y_true = df[true_col].astype(str)
        y_pred = df[pred_col].astype(str)

        m = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

        if "top10_preds" in df.columns:
            top10_acc = df.apply(
                lambda row: str(row[true_col]) in (row["top10_preds"] or [])[:10],
                axis=1
            ).mean()
            m["top10_acc"] = float(top10_acc)
        else:
            m["top10_acc"] = None

        return m

    # Domain metrics
    if df_domain is not None:
        domain_metrics = compute_metrics(df_domain, "top1_pred")
        if domain_metrics is None:
            print("[WARN] domain metrics unavailable.")

    # Duel metrics
    if df_duel is not None:
        duel_pred_col = "final_duel_pred" if "final_duel_pred" in df_duel.columns else (
            "duel_pred" if "duel_pred" in df_duel.columns else None
        )
        duel_metrics = compute_metrics(df_duel, duel_pred_col)
        if duel_metrics is None:
            print("[WARN] duel metrics unavailable.")

    # Plot (if both available)
    if domain_metrics and duel_metrics and domain_metrics.get("top10_acc") is not None and duel_metrics.get("top10_acc") is not None:
        categories = ["Accuracy Top-1", "Top-10 Accuracy", "F1 Macro"]
        fig = go.Figure(data=[
            go.Bar(name="Domain Expert", x=categories, y=[domain_metrics["accuracy"], domain_metrics["top10_acc"], domain_metrics["f1"]]),
            go.Bar(name="Duel Expert", x=categories, y=[duel_metrics["accuracy"], duel_metrics["top10_acc"], duel_metrics["f1"]]),
        ])
        fig.update_layout(title="📊 Comparaison Domain vs Duel Expert", barmode="group", template="plotly_white", height=380)
        plot_metrics_html = pio.to_html(fig, full_html=False)
    else:
        plot_metrics_html = None


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/ping")
def ping():
    return "pong", 200


@app.route("/health")
def health():
    files = []
    try:
        if os.path.exists(DATA_DIR):
            files = os.listdir(DATA_DIR)
    except Exception:
        pass

    return jsonify({
        "ok": True,
        "device": DEVICE,
        "data_dir": DATA_DIR,
        "data_files": files,
        "exists": {
            "global_csv": os.path.exists(GLOBAL_CSV),
            "domain_csv": os.path.exists(DOMAIN_CSV),
            "duel_csv": os.path.exists(DUEL_CSV),
            "pth_present": os.path.exists(PHASE2_CKPT),
            "static_bg": os.path.exists(os.path.join(STATIC_DIR, BACKGROUND_IMAGE_FILE))
        },
        "paths": {
            "GLOBAL_CSV": GLOBAL_CSV,
            "DOMAIN_CSV": DOMAIN_CSV,
            "DUEL_CSV": DUEL_CSV,
            "PHASE2_CKPT": PHASE2_CKPT
        },
        "env": {
            "HF_REPO_ID": HF_REPO_ID,
            "HF_FILENAME": HF_FILENAME,
            "HF_TOKEN_set": bool(HF_TOKEN),
            "HF_CACHE_DIR": HF_CACHE_DIR
        }
    })


@app.route("/health_model")
def health_model():
    """Debug endpoint: actually tries to run a tiny prediction."""
    try:
        labels, probs = predict_topk("test produit", k=3)
        return jsonify({"ok": True, "labels": labels, "conf": probs, "device": DEVICE})
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {str(e)}"}), 500


@app.route("/predict", methods=["POST"])
def predict_api():
    """
    JSON API:
    POST /predict
    body: {"text": "..." , "k": 10}
    returns JSON always (fixes front error: Unexpected end of JSON input)
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    k = data.get("k", 10)

    if not text:
        return jsonify({"ok": False, "error": "Missing 'text'"}), 400

    try:
        labels, probs = predict_topk(text, k=int(k))
        return jsonify({
            "ok": True,
            "top1": labels[0],
            "top10": labels,
            "conf": float(probs[0])
        })
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {str(e)}"}), 500


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_results = []
    llm_msg = ""
    start_time = time.time()

    background_image_url = url_for("static", filename=BACKGROUND_IMAGE_FILE)

    # load optional CSV + metrics
    try:
        load_optional_assets_if_needed()
    except Exception as e:
        print("[WARN] optional assets failed:", type(e).__name__, str(e))

    products = []
    if request.method == "POST":
        uploaded_file = request.files.get("csv_file")
        uploaded_image = request.files.get("product_image")
        new_product = (request.form.get("product_text") or "").strip()

        if uploaded_file:
            try:
                df_new = pd.read_csv(uploaded_file, sep=None, engine="python", on_bad_lines="skip")
                df_new = normalize_columns(df_new)

                if "description" in df_new.columns and "text" not in df_new.columns:
                    df_new.rename(columns={"description": "text"}, inplace=True)

                if "text" not in df_new.columns:
                    llm_msg = "CSV doit contenir une colonne 'text' (ou 'description')."
                else:
                    products = df_new["text"].astype(str).fillna("").tolist()
            except Exception as e:
                llm_msg = f"Erreur lecture CSV: {str(e)}"

        elif new_product:
            products = [new_product]

        elif uploaded_image:
            products = ["Description générée automatiquement à partir de l'image"]

    # Predict only if needed
    if products:
        try:
            for prod in products:
                prod = str(prod).strip()
                if not prod:
                    continue

                top_labels, top_probs = predict_topk(prod, k=10)
                prediction_results.append({
                    "text": prod,
                    "top1": top_labels[0],
                    "final": top_labels[0],
                    "top10": top_labels[:10],
                    "conf": round(float(top_probs[0]), 4)
                })
            llm_msg = "✅ Prédiction OK"
        except Exception as e:
            llm_msg = f"❌ Erreur modèle: {type(e).__name__}: {str(e)}"

    elapsed_time = round(time.time() - start_time, 2)

    example_products = df_duel.head(10).to_dict(orient="records") if df_duel is not None else []
    for p in example_products:
        p.setdefault("description", p.get("text", ""))
        p.setdefault("top1_pred", "-")
        p.setdefault("top10_preds", [])
        p.setdefault("final_duel_pred", p.get("duel_pred", "-"))
        p.setdefault("true_label", "-")

    return render_template(
        "domain_expert.html",
        background_image_url=background_image_url,
        domain_metrics=domain_metrics,
        duel_metrics=duel_metrics,
        products=example_products,
        plot_html=plot_metrics_html,
        prediction_results=prediction_results,
        llm_msg=llm_msg,
        elapsed_time=elapsed_time
    )


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
