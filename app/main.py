from typing import Optional
from fastapi import FastAPI, Response, status, HTTPException
from fastapi.params import Body
from pydantic import BaseModel

app = FastAPI()


class ImageInput(BaseModel):
    data: dict


@app.post("/api/models/pneumonia-detector/predict")
def predict_disease(input: ImageInput):
    return {"pneumonia-predictor": input}
