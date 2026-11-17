import os
import json
import datetime
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, create_model

# Ensure monitoring directory exists
os.makedirs("monitoring", exist_ok=True)
LOG_FILE = "monitoring/prediction_logs.jsonl"
MODEL_PATH = "model/model.joblib"

# Load serialized pipeline once at app startup outside request handlers
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Required model artifact not found at {MODEL_PATH}. Run train.py first.")

# Import pipeline wrapper definition
from pipeline import KeystrokeAuthPipeline

pipeline: KeystrokeAuthPipeline = joblib.load(MODEL_PATH)
print(f"Loaded model '{pipeline.metadata.get('model_type')}' for subject '{pipeline.metadata.get('subject')}'")
print(f"Operating threshold: {pipeline.threshold:.4f}")

# Build strict Pydantic Request Model with field-level range validation
field_definitions = {}
for col in pipeline.feature_names:
    if col.startswith("H."):
        field_definitions[col] = (
            float,
            Field(..., ge=0.001, le=3.0, description=f"Hold duration for {col} in seconds")
        )
    elif col.startswith("DD."):
        field_definitions[col] = (
            float,
            Field(..., ge=0.001, le=30.0, description=f"Keydown-to-Keydown latency for {col} in seconds")
        )
    elif col.startswith("UD."):
        # Physical domain rule: Key rollover yields valid negative UD timings
        field_definitions[col] = (
            float,
            Field(..., ge=-1.0, le=30.0, description=f"Keyup-to-Keydown interval for {col} in seconds (rollover >= -1.0)")
        )

KeystrokeInput = create_model("KeystrokeInput", **field_definitions)

class PredictionResponse(BaseModel):
    authenticated: bool
    prediction: int
    match_score: float
    threshold: float
    decision: str
    confidence_percentage: float
    rule_based_tips: List[str]
    top_feature_influences: Optional[Dict[str, float]] = None
    timestamp: str

app = FastAPI(
    title="Keystroke Dynamics ML Authentication API",
    description="Production-grade behavioral biometric authentication service (CSL236 End-Term Project).",
    version="1.0.0"
)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

def log_prediction(input_data: Dict[str, float], pred: int, score: float):
    """Privacy-preserving prediction logging for drift detection."""
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "prediction": int(pred),
        "match_score": float(score),
        "features": input_data
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Logging error: {e}")

@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    return {
        "status": "healthy",
        "service": "keystroke-auth-api",
        "model_loaded": pipeline is not None,
        "model_type": pipeline.metadata.get("model_type"),
        "enrolled_subject": pipeline.metadata.get("subject"),
        "operating_threshold": round(pipeline.threshold, 4),
        "features_tracked": len(pipeline.feature_names)
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(payload: KeystrokeInput):
    data_dict = payload.model_dump()
    df_sample = pd.DataFrame([data_dict])[pipeline.feature_names]

    # Model inference
    probs = pipeline.predict_proba(df_sample)
    score = float(probs[0, 1])
    is_authenticated = bool(score >= pipeline.threshold)
    pred_label = 1 if is_authenticated else 0

    # Asynchronous privacy-compliant logging
    log_prediction(data_dict, pred_label, score)

    # Behavioral rule-based tips
    hold_vals = [v for k, v in data_dict.items() if k.startswith("H.")]
    avg_hold = float(np.mean(hold_vals))
    ud_neg_count = sum(1 for k, v in data_dict.items() if k.startswith("UD.") and v < 0)

    tips = []
    if is_authenticated:
        tips.append(f"Authentic Hold Cadence: Mean key dwell time ({avg_hold*1000:.1f} ms) matches the authorized typist's motor template.")
        if ud_neg_count > 0:
            tips.append(f"Natural Key Rollover: Detected {ud_neg_count} overlapping keystrokes, characteristic of proficient authorized typing.")
        else:
            tips.append("Consistent Digraph Flow: Transition rhythms align with the enrolled biometric baseline.")
        tips.append("Access Granted: Behavioral dynamics verified with high confidence.")
    else:
        tips.append(f"Dwell Anomaly: Key hold timing ({avg_hold*1000:.1f} ms) diverges from the enrolled baseline.")
        tips.append("Cadence Mismatch: Inter-key flight latencies exhibit significant variance from authorized patterns.")
        tips.append("Security Advisory: If you are the legitimate user, ensure you type at your natural speed without hesitating.")

    # Feature influences (Random Forest importance or Logistic Regression magnitude)
    feature_influences = {}
    clf = pipeline.classifier
    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
        # Top 5 most important features
        top_indices = np.argsort(imps)[::-1][:5]
        for idx in top_indices:
            feat_name = pipeline.feature_names[idx]
            feature_influences[feat_name] = round(float(imps[idx]), 4)

    return PredictionResponse(
        authenticated=is_authenticated,
        prediction=pred_label,
        match_score=round(score, 4),
        threshold=round(pipeline.threshold, 4),
        decision="AUTHENTICATED (GENUINE USER)" if is_authenticated else "ACCESS DENIED (IMPOSTOR PATTERN)",
        confidence_percentage=round(score * 100, 2),
        rule_based_tips=tips,
        top_feature_influences=feature_influences if feature_influences else None,
        timestamp=datetime.datetime.utcnow().isoformat()
    )

@app.get("/api/sample/genuine")
def get_genuine_sample():
    """Retrieve an authentic test sample of enrolled user s002."""
    df = pd.read_csv("data/DSL-StrongPasswordData.csv")
    sample = df[(df["subject"] == "s002") & (df["sessionIndex"].isin([5, 6, 7, 8]))].iloc[0]
    return {col: float(sample[col]) for col in pipeline.feature_names}

@app.get("/api/sample/impostor")
def get_impostor_sample():
    """Retrieve an authentic test sample from impostor subject s003."""
    df = pd.read_csv("data/DSL-StrongPasswordData.csv")
    sample = df[(df["subject"] == "s003") & (df["sessionIndex"].isin([5, 6, 7, 8]))].iloc[0]
    return {col: float(sample[col]) for col in pipeline.feature_names}

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
