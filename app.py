from fastapi import FastAPI
from pydantic import BaseModel
from yield_model import predict_yield

class YieldInput(BaseModel):
    Region: str
    Soil_Type: str
    Crop: str
    Weather_Condition: str
    Rainfall_mm: float
    Temperature_Celsius: float
    Days_to_Harvest: int
    NDVI: float
    EVI: float

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Smart Farm Yield Prediction API is running"}

@app.post("/predict")
def predict(input_data: YieldInput):
    data_dict = input_data.dict()
    prediction = predict_yield(data_dict)
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

