# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import numpy as np
# import pandas as pd
# from typing import Dict

# import os
# import xgboost as xgb
# from typing import Optional

# from rag_module import initialize_system, rag_query

# # Initialize FastAPI
# app = FastAPI(title="Pregnancy Risk Prediction API with RAG", version="2.0.0")

# # === Load ML model and encoders ===
# try:
#     model = joblib.load('stats/xgb_model.pkl')
#     le_y = joblib.load('stats/label_encoder_y.pkl')
#     label_encoders = joblib.load('stats/label_encoders_features.pkl')

#     if isinstance(model, xgb.XGBClassifier):
#         model_features = model.get_booster().feature_names
#     else:
#         model_features = list(label_encoders.keys()) + [
#             'BP', 'BP1', 'HEMOGLOBIN', 'HEART_RATE', 'BLOOD_SUGAR', 'FEVER',
#             'IFA_QUANTITY', 'NO_OF_WEEKS', 'PHQ_SCORE', 'GAD_SCORE',
#             'PULSE_RATE', 'RESPIRATORY_RATE', 'UTERUS_SIZE', 'OGTT_2_HOURS',
#             'KNOWN_EPILEPTIC', 'CONVULSION_SEIZURES', 'FOLIC_ACID',
#             'AGE', 'HEIGHT', 'WEIGHT'
#         ]
# except Exception as e:
#     raise RuntimeError(f"Failed to load ML model or encoders: {str(e)}")

# # === Load RAG system ===
# rag_model, rag_client = initialize_system(use_gpu=False)

# # === Risk level mapping ===
# output_mapping = {
#     "low risk": 1,
#     "medium risk": 2,
#     "high risk": 3
# }
# reverse_mapping = {
#     1: "Low Risk",
#     2: "Medium Risk",
#     3: "High Risk"
# }

# # === Input/Output Schemas ===
# class PredictionInput(BaseModel):
#     AGE: float
#     HEIGHT: float
#     WEIGHT: float
#     BLOOD_GRP: str
#     HUSBAND_BLOOD_GROUP: str
#     GRAVIDA: str
#     PARITY: str
#     ABORTIONS: str
#     PREVIOUS_ABORTION: str
#     LIVE: str
#     DEATH: str
#     KNOWN_EPILEPTIC: float
#     TWIN_PREGNANCY: str
#     GESTANTIONAL_DIA: str
#     CONVULSION_SEIZURES: float
#     BP: float
#     BP1: float
#     HEMOGLOBIN: float
#     PULSE_RATE: float
#     RESPIRATORY_RATE: float
#     HEART_RATE: float
#     FEVER: float
#     OEDEMA: str
#     OEDEMA_TYPE: str
#     UTERUS_SIZE: float
#     URINE_SUGAR: str
#     URINE_ALBUMIN: str
#     THYROID: str
#     RH_NEGATIVE: str
#     SYPHYLIS: str
#     HIV: str
#     HIV_RESULT: str
#     HEP_RESULT: str
#     BLOOD_SUGAR: float
#     OGTT_2_HOURS: float
#     WARNING_SIGNS_SYMPTOMS_HTN: str
#     ANY_COMPLAINTS_BLEEDING_OR_ABNORMAL_DISCHARGE: str
#     IFA_TABLET: str
#     IFA_QUANTITY: float
#     IRON_SUCROSE_INJ: str
#     CALCIUM: str
#     FOLIC_ACID: float
#     SCREENED_FOR_MENTAL_HEALTH: str
#     PHQ_SCORE: float
#     GAD_SCORE: float
#     PHQ_ACTION: str
#     GAD_ACTION: str
#     ANC1FLG: str
#     ANC2FLG: str
#     ANC3FLG: str
#     ANC4FLG: str
#     MISSANC1FLG: str
#     MISSANC2FLG: str
#     MISSANC3FLG: str
#     MISSANC4FLG: str
#     NO_OF_WEEKS: float
#     DELIVERY_MODE: str
#     PLACE_OF_DELIVERY: str
#     IS_PREV_PREG: str
#     CONSANGUINITY: str

# class PredictionOutput(BaseModel):
#     ml_risk: str
#     rag_risk: str
#     combined_risk: str
#     confidence: float
#     probabilities: Dict[str, float]

# # === ML Prediction function ===
# def classify_risk(input_data: Dict) -> (str, float, Dict[str, float]):
#     df = pd.DataFrame([input_data])
    
#     # Clean numeric
#     numeric_cols = ['BP', 'BP1', 'HEMOGLOBIN', 'HEART_RATE', 'BLOOD_SUGAR', 'FEVER',
#                     'IFA_QUANTITY', 'NO_OF_WEEKS', 'PHQ_SCORE', 'GAD_SCORE',
#                     'PULSE_RATE', 'RESPIRATORY_RATE', 'UTERUS_SIZE', 'OGTT_2_HOURS',
#                     'KNOWN_EPILEPTIC', 'CONVULSION_SEIZURES', 'FOLIC_ACID',
#                     'AGE', 'HEIGHT', 'WEIGHT']
#     for col in numeric_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())

#     # Label encode
#     for col, le in label_encoders.items():
#         if col in df.columns:
#             df[col] = df[col].astype(str)
#             unseen_mask = ~df[col].isin(le.classes_)
#             if unseen_mask.any():
#                 df.loc[unseen_mask, col] = le.classes_[0]
#             df[col] = le.transform(df[col])

#     # Fill and reorder
#     for col in set(model_features) - set(df.columns):
#         df[col] = 0
#     df = df[model_features]

#     probs = model.predict_proba(df)[0]
#     pred_idx = np.argmax(probs)
#     pred_label = le_y.inverse_transform([pred_idx])[0]
#     confidence = probs[pred_idx]
#     prob_dict = {le_y.classes_[i]: float(p) for i, p in enumerate(probs)}

#     return pred_label, float(confidence), prob_dict
# class PredictionOutput(BaseModel):
#     ml_risk: Optional[str]
#     rag_risk: str
#     combined_risk: str
#     confidence: Optional[float]
#     probabilities: Optional[Dict[str, float]]
# # === Endpoints ===
# @app.get("/")
# def root():
#     return {"message": "Pregnancy Risk ML + RAG API", "version": "2.0"}

# @app.post("/combined_predict", response_model=PredictionOutput)
# def combined_predict(input_data: PredictionInput):
#     try:
#         input_dict = input_data.dict()
#         # ML model
#         ml_label, confidence, prob_dict = classify_risk(input_dict)
#         ml_clean = ml_label.lower().strip().replace("_", " ")

#         # RAG
#         rag_raw = rag_query(rag_model, rag_client, str(input_dict))
#         rag_clean = rag_raw.lower().strip().replace("_", " ")

#         ml_score = output_mapping.get(ml_clean, 1)
#         rag_score = output_mapping.get(rag_clean, 1)
#         combined_score = round((ml_score + rag_score) / 2)
#         combined_risk = reverse_mapping.get(combined_score, "Medium Risk")

#         return {
#             # "ml_risk": ml_label,
#             "rag_risk": rag_raw,
#             "combined_risk": combined_risk,
#             "confidence": confidence,
#             # "probabilities": prob_dict
#         }

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# # === Run as script ===
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("combined_output:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional
import re 
import os
import xgboost as xgb

# from rag_module import initialize_system, rag_query
from fastapi.responses import JSONResponse
from rag_chat import initialize_system, rag_query

# Initialize FastAPI
app = FastAPI(title="Pregnancy Risk Prediction API with RAG", version="1.0.0")

# === Load ML model and encoders ===
try:
    model = joblib.load('stats/xgb_model.pkl')
    le_y = joblib.load('stats/label_encoder_y.pkl')
    label_encoders = joblib.load('stats/label_encoders_features.pkl')

    if isinstance(model, xgb.XGBClassifier):
        model_features = model.get_booster().feature_names
    else:
        model_features = list(label_encoders.keys()) + [
            'BP', 'BP1', 'HEMOGLOBIN', 'HEART_RATE', 'BLOOD_SUGAR', 'FEVER',
            'IFA_QUANTITY', 'NO_OF_WEEKS', 'PHQ_SCORE', 'GAD_SCORE',
            'PULSE_RATE', 'RESPIRATORY_RATE', 'UTERUS_SIZE', 'OGTT_2_HOURS',
            'KNOWN_EPILEPTIC', 'CONVULSION_SEIZURES', 'FOLIC_ACID',
            'AGE', 'HEIGHT', 'WEIGHT'
        ]
except Exception as e:
    raise RuntimeError(f"Failed to load ML model or encoders: {str(e)}")

# === Load RAG system ===
rag_model, rag_client = initialize_system(use_gpu=False)

# === Risk level mapping ===
output_mapping = {
    "low risk": 1,
    "medium risk": 2,
    "high risk": 3
}
reverse_mapping = {
    1: "Low Risk",
    2: "Medium Risk",
    3: "High Risk"
}

# === Input Schema ===
class PredictionInput(BaseModel):
    AGE: float
    HEIGHT: float
    WEIGHT: float
    BLOOD_GRP: str
    HUSBAND_BLOOD_GROUP: str
    GRAVIDA: str
    PARITY: str
    ABORTIONS: str
    PREVIOUS_ABORTION: str
    LIVE: str
    DEATH: str
    KNOWN_EPILEPTIC: float
    TWIN_PREGNANCY: str
    GESTANTIONAL_DIA: str
    CONVULSION_SEIZURES: float
    BP: float
    BP1: float
    HEMOGLOBIN: float
    PULSE_RATE: float
    RESPIRATORY_RATE: float
    HEART_RATE: float
    FEVER: float
    OEDEMA: str
    OEDEMA_TYPE: str
    UTERUS_SIZE: float
    URINE_SUGAR: str
    URINE_ALBUMIN: str
    THYROID: str
    RH_NEGATIVE: str
    SYPHYLIS: str
    HIV: str
    HIV_RESULT: str
    HEP_RESULT: str
    BLOOD_SUGAR: float
    OGTT_2_HOURS: float
    WARNING_SIGNS_SYMPTOMS_HTN: str
    ANY_COMPLAINTS_BLEEDING_OR_ABNORMAL_DISCHARGE: str
    IFA_TABLET: str
    IFA_QUANTITY: float
    IRON_SUCROSE_INJ: str
    CALCIUM: str
    FOLIC_ACID: float
    SCREENED_FOR_MENTAL_HEALTH: str
    PHQ_SCORE: float
    GAD_SCORE: float
    PHQ_ACTION: str
    GAD_ACTION: str
    ANC1FLG: str
    ANC2FLG: str
    ANC3FLG: str
    ANC4FLG: str
    MISSANC1FLG: str
    MISSANC2FLG: str
    MISSANC3FLG: str
    MISSANC4FLG: str
    NO_OF_WEEKS: float
    DELIVERY_MODE: str
    PLACE_OF_DELIVERY: str
    IS_PREV_PREG: str
    CONSANGUINITY: str

# === Output Schema ===
class PredictionOutput(BaseModel):
    rag_risk: str
    combined_risk: str
    confidence: Optional[float] = None
    ml_risk: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None

# === ML Prediction Function ===
def classify_risk(input_data: Dict) -> (str, float, Dict[str, float]):
    df = pd.DataFrame([input_data])
    numeric_cols = ['BP', 'BP1', 'HEMOGLOBIN', 'HEART_RATE', 'BLOOD_SUGAR', 'FEVER',
                    'IFA_QUANTITY', 'NO_OF_WEEKS', 'PHQ_SCORE', 'GAD_SCORE',
                    'PULSE_RATE', 'RESPIRATORY_RATE', 'UTERUS_SIZE', 'OGTT_2_HOURS',
                    'KNOWN_EPILEPTIC', 'CONVULSION_SEIZURES', 'FOLIC_ACID',
                    'AGE', 'HEIGHT', 'WEIGHT']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())

    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            unseen_mask = ~df[col].isin(le.classes_)
            if unseen_mask.any():
                df.loc[unseen_mask, col] = le.classes_[0]
            df[col] = le.transform(df[col])

    for col in set(model_features) - set(df.columns):
        df[col] = 0
    df = df[model_features]

    probs = model.predict_proba(df)[0]
    pred_idx = np.argmax(probs)
    pred_label = le_y.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx]
    prob_dict = {le_y.classes_[i]: float(p) for i, p in enumerate(probs)}

    return pred_label, float(confidence), prob_dict

# === Endpoints ===
@app.get("/")
def root():
    return {"message": "Pregnancy Risk ML + RAG API", "version": "1.0"}

@app.post("/combined_predict", response_model=PredictionOutput)
def combined_predict(input_data: PredictionInput):
    try:
        input_dict = input_data.dict()
        ml_label, confidence, prob_dict = classify_risk(input_dict)
        ml_clean = ml_label.lower().strip().replace("_", " ")

        # rag_raw = rag_query(rag_model, rag_client, str(input_dict))
        # rag_clean = rag_raw.lower().strip().replace("_", " ")
        rag_result = rag_query(rag_model, rag_client, str(input_dict))

        # Extract <think> explanation and one-word risk
        reasoning_match = re.search(r"<think>(.*?)</think>", rag_result, re.DOTALL)
        rag_explanation = reasoning_match.group(1).strip() if reasoning_match else ""
        rag_label = rag_result.split("</think>")[-1].strip().split()[0] if "</think>" in rag_result else rag_result.strip()
        rag_clean = rag_label.lower().strip().replace("_", " ")

        ml_score = output_mapping.get(ml_clean, 1)
        rag_score = output_mapping.get(rag_clean, 1)
        combined_score = round((ml_score + rag_score) / 2)
        combined_risk = reverse_mapping.get(combined_score, "Medium Risk")

        # Return only required fields, exclude unset/null ones
        response = PredictionOutput(
            rag_risk=rag_label,
            # rag_risk=rag_raw,
            combined_risk=combined_risk,
            confidence=confidence
        )
        return JSONResponse(content=response.dict(exclude_unset=True))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# === Run as Script ===
if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # 👈 Fix for Render
    uvicorn.run("combined_output:app", host="0.0.0.0", port=port, reload=True)
