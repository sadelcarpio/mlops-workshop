import pandas as pd
from fastapi import FastAPI
import mlflow
from pydantic import BaseModel

app = FastAPI()

mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.sklearn.load_model("models:/production-model/latest")


class Features(BaseModel):
    fixed: float
    acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.post("/predict")
def predict(data: Features):
    data = pd.DataFrame(data.dict(), index=[0])
    quality = model.predict(data)
    return {"predicted_quality": quality}
