# ================================
# app.py – Duel Expert Flask App (Render-friendly)
# ================================
from flask import Flask, render_template, request
import pandas as pd, ast, os, gc, time, re
import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel, pipeline
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objs as go
import plotly.io as pio

# ----------------------------
# CONFIG FLASK & PATHS RELATIFS
# ----------------------------
app = Flask(__name__)
BACKGROUND_IMAGE = "static/imageeco.jpg"

DOMAIN_CSV = "data/domain_expert_phase2_top10_predictions_full.csv"
DUEL_CSV = "data/duel_expert_mistral_GENERAL_EXPERT_top10_corrected.csv"
GLOBAL_CSV = "data/produits_nettoyes.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 160
NUM_CLASSES = 882

# ----------------------------
# DOMAIN EXPERT MODEL (Large ou Base)
# ----------------------------
class DomainExpert(nn.Module):
    def __init__(self, n_classes, base_model="xlm-roberta-base", dropout=0.3):
        super().__init__()
        self.xlm = XLMRobertaModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.xlm.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.xlm(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))

# ----------------------------
# TOKENIZER & LABEL ENCODER
# ----------------------------
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
df_global = pd.read_csv(GLOBAL_CSV, sep=';', on_bad_lines='skip')
df_global['taxonomy_path'] = df_global['taxonomy_path'].astype(str)
le = LabelEncoder()
le.fit(df_global['taxonomy_path'].tolist())

# ----------------------------
# DOMAIN EXPERT – Lazy loading
# ----------------------------
model = None

def load_domain_expert():
    global model
    if model is None:
        print("Chargement du DomainExpert…")
        # TEST RAM : si GPU ou trop peu de RAM → Base, sinon Large
        try:
            model = DomainExpert(NUM_CLASSES, base_model="xlm-roberta-large").to(DEVICE)
        except RuntimeError:
            print("RAM insuffisante pour Large, fallback sur Base")
            model = DomainExpert(NUM_CLASSES, base_model="xlm-roberta-base").to(DEVICE)

        try:
            ckpt_path = hf_hub_download(
                repo_id="Bineta123/domain-expert-xlmr",
                filename="domain_expert_flat.pth",
                token=os.environ.get("HF_TOKEN")  # utilise le token Hugging Face
            )
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(new_sd, strict=False)
            del ckpt, sd
            gc.collect()
            torch.cuda.empty_cache()
            print("DomainExpert chargé !")
        except Exception as e:
            print(f"Erreur chargement modèle Hugging Face : {e}")

# ----------------------------
# LLM API (HuggingFace) – Lazy loading avec token
# ----------------------------
llm_pipeline = None

def load_llm_api():
    global llm_pipeline
    if llm_pipeline is None:
        hf_token = os.environ.get("HF_TOKEN")
        llm_pipeline = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            device_map="auto",
            torch_dtype="auto",
            use_auth_token=hf_token
        )

# ----------------------------
# Fonction LLM correct
# ----------------------------
def llm_correct_api(text, top10_preds, wrong_top1):
    load_llm_api()
    prompt = f"""
You are a GENERAL EXPERT correcting a WRONG product classification.
IMPORTANT: The current top-1 prediction is WRONG and must NOT be selected again.

Product:
{text}

Candidate categories:
{', '.join(top10_preds)}

Rules:
- Choose exactly ONE category
- It must be DIFFERENT from the wrong top-1
- Choose only from the list
- No explanations

Final choice:
"""
    outputs = llm_pipeline(prompt, max_new_tokens=40, do_sample=True, temperature=0.3)
    out_text = outputs[0]["generated_text"]
    match = re.search(r"Final choice\s*:\s*(.+)", out_text, re.IGNORECASE)
    chosen = None
    if match:
        pred = match.group(1).strip()
        for c in top10_preds:
            if pred.lower() == c.lower():
                chosen = c
                break
    if chosen is None or chosen == wrong_top1:
        chosen = top10_preds[1] if len(top10_preds) > 1 else top10_preds[0]
    return chosen

# ----------------------------
# MAIN FLASK
# ----------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
