# app.py
# Run: uvicorn app:app --reload --port 8000

import pickle, yaml
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# โหลด config และโมเดล
with open("risk_config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

MODEL_PATH = CFG.get("model_path", "models/xgb_model.sav")
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

TH_LOW = float(CFG["thresholds"]["low"])
TH_HIGH = float(CFG["thresholds"]["high"])
MSG = CFG["messages"]

# หาลำดับฟีเจอร์จากโมเดล
try:
    FEATURES = list(MODEL.feature_names_in_)  # scikit-learn API
except Exception:
    try:
        FEATURES = list(MODEL.get_booster().feature_names)  # xgboost booster
    except Exception:
        FEATURES = [
            "gender","age_60_and_above","cough","fever","sore_throat",
            "shortness_of_breath","head_ache","abroad","contact_with_covid_positive_patient"
        ]

app = FastAPI(title="COVID Risk Predictor")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def predict_one(row: dict) -> dict:
    # เติม missing features = 0 และจัดเรียงตาม FEATURES
    all_cols = FEATURES
    df = pd.DataFrame([{c: int(row.get(c, 0)) for c in all_cols}], columns=all_cols)

    if hasattr(MODEL, "predict_proba"):
        p = float(MODEL.predict_proba(df)[:, 1][0])
    elif hasattr(MODEL, "decision_function"):
        z = float(MODEL.decision_function(df)[0]); p = 1/(1+np.exp(-z))
    else:
        p = float(MODEL.predict(df)[0])

    if p >= TH_HIGH:
        return {"probability": p, "risk_level": "ความเสี่ยงสูง", "recommendation": MSG["high"]}
    elif p < TH_LOW:
        return {"probability": p, "risk_level": "ความเสี่ยงต่ำ", "recommendation": MSG["low"]}
    else:
        return {"probability": p, "risk_level": "ความเสี่ยงปานกลาง", "recommendation": MSG["medium"]}


@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})


@app.post("/", response_class=HTMLResponse)
async def form_submit(
    request: Request,
    gender: int = Form(...),
    age: int = Form(...),  # <-- กรอกอายุจริง
    cough: int = Form(0),
    fever: int = Form(0),
    sore_throat: int = Form(0),
    shortness_of_breath: int = Form(0),
    head_ache: int = Form(0),
    abroad: int = Form(0),
    contact_with_covid_positive_patient: int = Form(0),

    # ====== เพิ่มฟีเจอร์ใหม่ ======
    chronic_disease: int = Form(0),
    previous_covid: int = Form(0),
    chills: int = Form(0),
    diarrhea: int = Form(0),
    loss_of_smell_taste: int = Form(0),
    vaccine_doses: int = Form(0),
):
    row = {
        "gender": gender,
        "age_60_and_above": 1 if age >= 60 else 0,
        "cough": cough,
        "fever": fever,
        "sore_throat": sore_throat,
        "shortness_of_breath": shortness_of_breath,
        "head_ache": head_ache,
        "abroad": abroad,
        "contact_with_covid_positive_patient": contact_with_covid_positive_patient,

        # ====== ฟีเจอร์ใหม่ ======
        "chronic_disease": chronic_disease,
        "previous_covid": previous_covid,
        "chills": chills,
        "diarrhea": diarrhea,
        "loss_of_smell_taste": loss_of_smell_taste,
        "vaccine_doses": vaccine_doses,
    }
    result = predict_one(row)
    return templates.TemplateResponse("form.html", {"request": request, "result": result})

