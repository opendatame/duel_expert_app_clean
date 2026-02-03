# ================================
# app.py – Duel Expert Flask App (Render-ready, CSV in GitHub, HF private .pth)
# ================================
from flask import Flask, render_template, request, jsonify
import os, time, ast, gc
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

BACKGROUND_IMAGE = os.getenv("BACKGROUND_IMAGE", os.path.join(STATIC_DIR, "imageeco.jpg"))

DOMAIN_CSV = os.getenv("DOMAIN_CSV", os.path.join(DATA_DIR, "domain_expert_phase2_top10_predictions_full.csv"))
DUEL_CSV   = os.getenv("DUEL_CSV",   os.path.join(DATA_DIR, "duel_expert_mistral_GENERAL_EXPERT_top10_corrected.csv"))
GLOBAL_CSV = os.getenv("GLOBAL_CSV", os.path.join(DATA_DIR, "produits_nettoyes.csv"))

PHASE2_CKPT = os.getenv("PHASE2_CKPT", os.path.join(MODEL_DIR, "domain_expert_flat.pth"))

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = int(os.getenv("MAX_LEN", "160"))


# ----------------------------
# HF PRIVATE MODEL DOWNLOAD (.pth)
# ----------------------------
HF_REPO_ID  = os.getenv("HF_REPO_ID", "Bineta123/domain-expert-xlmr")
HF_FILENAME = os.getenv("HF_FILENAME", "domain_expert_flat.pth")
HF_TOKEN    = os.getenv("HF_TOKEN")  # SECRET env var on Render

def ensure_model_file(local_path: str):
    """Download private .pth from HF if missing."""
    if os.path.exists(local_path):
        try:
            if os.path.getsize(local_path) > 10_000_000:
                print("[MODEL] already present:", local_path)
                return
        except Exception:
            pass

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN missing. Add it in Render → Environment → Secret."
        )

    print("[MODEL] downloading .pth from Hugging Face private repo...")
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        token=HF_TOKEN
    )

    import shutil
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copyfile(downloaded, local_path)
    print("[MODEL] ✅ downloaded to:", local_path)


# ----------------------------
# SAFE CSV READER
# ----------------------------
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

    # Try common separators
    for sep in [";", ",", None]:
        try:
            if sep is None:
                df = pd.read_csv(path, on_bad_lines="skip")
            else:
                df = pd.read_csv(path, sep=sep, on_bad_lines="skip")

            if df is None or df.shape[1] == 0:
                continue
            return df
        except pd.errors.EmptyDataError:
            return None
        except Exception:
            continue

    return None


# ----------------------------
# DOMAIN EXPERT MODEL
# ----------------------------
class DomainExpert(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        # IMPORTANT: no local_files_only on Render
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

    if not os.path.exists(GLOBAL_CSV):
        raise FileNotFoundError(
            f"GLOBAL_CSV missing: {GLOBAL_CSV} (put produits_nettoyes.csv in data/)"
        )

    df_global = safe_read_csv(GLOBAL_CSV)
    if df_global is None:
        raise RuntimeError("GLOBAL_CSV is empty or unreadable.")

    if "taxonomy_path" not in df_global.columns:
        raise ValueError("GLOBAL_CSV must contain column 'taxonomy_path'")

    df_global["taxonomy_path"] = df_global["taxonomy_path"].astype(str)

    le_local = LabelEncoder()
    le_local.fit(df_global["taxonomy_path"].tolist())

    le = le_local
    NUM_CLASSES = len(le.classes_)
    print("[INIT] NUM_CLASSES:", NUM_CLASSES)


def extract_state_dict(ckpt):
    """Support checkpoints: raw state_dict OR dict with model_state_dict/state_dict."""
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

    print("[LOAD] model weights...")
    model_local = DomainExpert(NUM_CLASSES).to(DEVICE)

    ckpt = torch.load(PHASE2_CKPT, map_location=DEVICE)
    sd = extract_state_dict(ckpt)

    if isinstance(sd, dict):
        # remove DataParallel prefix if any
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

    model_local.load_state_dict(sd, strict=False)
    model_local.eval()

    # cleanup
    del ckpt, sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = model_local
    print("[LOAD] ✅ model ready.")


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

    logits = model(input_ids=input_ids, attention_mask=attention_mask)  # [1, C]
    probs = F.softmax(logits, dim=-1).squeeze(0)                        # [C]

    kk = max(1, min(int(k), int(probs.shape[0])))
    top_probs, top_idx = torch.topk(probs, k=kk)

    labels = le.inverse_transform(top_idx.detach().cpu().numpy()).tolist()
    probs_list = top_probs.detach().cpu().numpy().tolist()
    return labels, probs_list


def load_optional_assets_if_needed():
    """Load domain/duel CSV + metrics if available (never crash the app)."""
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

    if df_domain is None or df_duel is None:
        domain_metrics = None
        duel_metrics = None
        plot_metrics_html = None
        return

    try:
        def compute_top1_metrics(df, pred_col):
            y_true = df["true_label"]
            y_pred = df[pred_col]
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            }

        def compute_top10_accuracy(df, top10_col):
            return df.apply(lambda row: row["true_label"] in (row[top10_col] or [])[:10], axis=1).mean()

        domain_metrics = compute_top1_metrics(df_domain, "top1_pred")
        domain_metrics["top10_acc"] = compute_top10_accuracy(df_domain, "top10_preds")

        duel_metrics = compute_top1_metrics(df_duel, "final_duel_pred")
        duel_metrics["top10_acc"] = compute_top10_accuracy(df_duel, "top10_preds")

        categories = ["Accuracy Top-1", "Top-10 Accuracy", "F1 Macro"]
        fig = go.Figure(data=[
            go.Bar(name="Domain Expert", x=categories, y=[domain_metrics["accuracy"], domain_metrics["top10_acc"], domain_metrics["f1"]]),
            go.Bar(name="Duel Expert", x=categories, y=[duel_metrics["accuracy"], duel_metrics["top10_acc"], duel_metrics["f1"]]),
        ])
        fig.update_layout(
            title="📊 Comparaison Domain vs Duel Expert",
            barmode="group",
            template="plotly_white",
            height=380
        )
        plot_metrics_html = pio.to_html(fig, full_html=False)

    except Exception as e:
        print("[WARN] metrics disabled:", type(e).__name__, str(e))
        domain_metrics = None
        duel_metrics = None
        plot_metrics_html = None


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/ping")
def ping():
    return "pong", 200


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "device": DEVICE,
        "paths": {
            "GLOBAL_CSV": GLOBAL_CSV,
            "DOMAIN_CSV": DOMAIN_CSV,
            "DUEL_CSV": DUEL_CSV,
            "PHASE2_CKPT": PHASE2_CKPT,
        },
        "exists": {
            "global_csv": os.path.exists(GLOBAL_CSV),
            "domain_csv": os.path.exists(DOMAIN_CSV),
            "duel_csv": os.path.exists(DUEL_CSV),
            "pth_present": os.path.exists(PHASE2_CKPT),
        },
        "env": {
            "HF_REPO_ID": HF_REPO_ID,
            "HF_FILENAME": HF_FILENAME,
            "HF_TOKEN_set": bool(HF_TOKEN),
        }
    })


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_results = []
    llm_msg = ""
    start_time = time.time()

    # Load optional CSV + metrics for display (won't crash)
    try:
        load_optional_assets_if_needed()
    except Exception:
        pass

    products = []
    if request.method == "POST":
        uploaded_file = request.files.get("csv_file")
        uploaded_image = request.files.get("product_image")
        new_product = (request.form.get("product_text") or "").strip()

        if uploaded_file:
            try:
                df_new = pd.read_csv(uploaded_file, sep=";", on_bad_lines="skip")
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
            # placeholder: you can plug a real image caption model later
            products = ["Description générée automatiquement à partir de l'image"]

    # Only load model if needed
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
                    "final": top_labels[0],  # duel step disabled here
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
        p.setdefault("final_duel_pred", "-")
        p.setdefault("true_label", "-")

    return render_template(
        "domain_expert.html",
        background_image=BACKGROUND_IMAGE,
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
