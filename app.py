import os
import sys
import traceback
import requests
import pandas as pd
import torch

from flask import Flask, render_template, request, jsonify
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from sklearn.preprocessing import LabelEncoder

# =========================================================
# CONFIGURATION DES PATHS (PORTABLE)
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

BACKGROUND_IMAGE = os.path.join(STATIC_DIR, "imageeco.jpg")

GLOBAL_CSV = os.path.join(DATA_DIR, "produits_nettoyes.csv")

PHASE2_CKPT = os.path.join(MODEL_DIR, "domain_expert_flat.pth")

# =========================================================
# MODEL DOWNLOAD CONFIG
# =========================================================

MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/TON_USER/domain-expert/resolve/main/domain_expert_flat.pth"
)

def download_model_if_needed():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(PHASE2_CKPT):
        print("[DOWNLOAD] Téléchargement du modèle...")
        r = requests.get(MODEL_URL, stream=True, timeout=120)
        r.raise_for_status()

        with open(PHASE2_CKPT, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("[DOWNLOAD] ✅ Modèle téléchargé")

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# GLOBAL OBJECTS
# =========================================================

tokenizer = None
encoder   = None
model     = None

MODEL_READY = False
MODEL_ERROR = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# MODEL DEFINITION (EXEMPLE)
# =========================================================

class DomainExpertModel(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)

# =========================================================
# LOAD EVERYTHING (SAFE)
# =========================================================

def load_everything():
    global tokenizer, encoder, model, MODEL_READY, MODEL_ERROR

    if MODEL_READY:
        return

    try:
        print("[INIT] Initialisation du système...")

        # 1️⃣ Télécharger le modèle si nécessaire
        download_model_if_needed()

        # 2️⃣ Charger les données
        if not os.path.exists(GLOBAL_CSV):
            raise FileNotFoundError(f"CSV manquant : {GLOBAL_CSV}")

        df = pd.read_csv(GLOBAL_CSV)

        if "label" not in df.columns or "text" not in df.columns:
            raise ValueError("Le CSV doit contenir les colonnes 'text' et 'label'")

        # 3️⃣ Encoder labels
        encoder = LabelEncoder()
        encoder.fit(df["label"].astype(str))
        num_labels = len(encoder.classes_)

        # 4️⃣ Tokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

        # 5️⃣ Modèle
        model = DomainExpertModel(num_labels=num_labels)
        state = torch.load(PHASE2_CKPT, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()

        MODEL_READY = True
        print("[INIT] ✅ Modèle prêt")

    except Exception as e:
        MODEL_ERROR = str(e)
        print("[ERROR] ❌ Erreur chargement modèle")
        traceback.print_exc()

# =========================================================
# INFERENCE
# =========================================================

def predict(text: str):
    if not MODEL_READY:
        raise RuntimeError("Modèle non prêt")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs)

    probs = torch.softmax(logits, dim=1)[0]
    idx = torch.argmax(probs).item()

    return {
        "label": encoder.inverse_transform([idx])[0],
        "confidence": float(probs[idx])
    }

# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def index():
    return render_template("index.html", background=BACKGROUND_IMAGE)

@app.route("/predict", methods=["POST"])
def predict_route():
    if not MODEL_READY:
        return jsonify({"error": MODEL_ERROR or "Modèle non chargé"}), 500

    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Texte vide"}), 400

    result = predict(text)
    return jsonify(result)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    load_everything()
    app.run(host="0.0.0.0", port=5000, debug=True)
