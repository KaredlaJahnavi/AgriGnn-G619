from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import yield_model
import threading

app = FastAPI(title="Crop Yield Prediction API")

class PredictInput(BaseModel):
    Region: str
    Soil_Type: str
    Crop: str
    Weather_Condition: str
    Rainfall_mm: float
    Temperature_Celsius: float
    Days_to_Harvest: float
    NDVI: float
    EVI: float
# ✅ Add these imports
import torch
import numpy as np
import random
import os

# ✅ Set random seeds for full reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ✅ For CUDA determinism (if GPU is used)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Initialize model components at startup (runs once)
_components = {"ready": False}
start_lock = threading.Lock()

@app.on_event("startup")
def load_model():
    with start_lock:
        if _components["ready"]:
            return
        try:
            df = yield_model.load_full_dataset()
            if df is None:
                raise Exception("No dataset file found.")
            comps = yield_model.setup_full_gnn(df)
            _components.update(comps)
            _components["ready"] = True
            print("✅ Model and preprocessing loaded for FastAPI.")
        except Exception as e:
            print(str(e))
            raise

@app.post("/predict")
def predict_yield(input: PredictInput):
    if not _components["ready"]:
        raise HTTPException(status_code=503, detail="Model not ready yet; try again in a moment.")
    user_data = input.dict()
    try:
        pred, confidence, accuracy = yield_model.predict_with_full_gnn(user_data, _components)
        return {"prediction": float(pred), "confidence": int(confidence), "accuracy": int(accuracy)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
