# fmt: off
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from PIL import Image, ImageOps
from io import BytesIO
from keras.models import load_model
import numpy as np
import cv2
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
import imghdr
# fmt: on


# get the absolute path of the directory where this file is located
dir_path = os.path.dirname(os.path.realpath(__file__))
model_dir_path = os.path.join(dir_path, "../models")

app = FastAPI()


np.set_printoptions(suppress=True)  # suppress scientific notations

# Load Models
pneumoniaDetectorModel = load_model(
    f"{model_dir_path}/pneumonia-predictor/pneumonia-predictor.h5")
pneumoniaClassNames = [line.rstrip('\n') for line in open(
    f"{model_dir_path}/pneumonia-predictor/labels.txt", "r")]


def inference_pneumonia_predictor(image):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = pneumoniaDetectorModel.predict(data)
    index = np.argmax(prediction)
    class_name = pneumoniaClassNames[index]
    confidence_score = prediction[0][index]
    return {"prediction": f"{class_name}", "confidence": f"{confidence_score}"}


@app.post("/api/models/pneumonia-predictor/predict")
async def predict_disease(file: UploadFile = File(...)):
    valid_image_types = ["jpg", "jpeg", "png", "gif"]
    file_type = imghdr.what(file.file)
    # check if the file is an image
    if file_type not in valid_image_types:
        raise HTTPException(
            status_code=400, detail="Only image files are allowed.")

    # read the image file using pillow
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    results = inference_pneumonia_predictor(image)
    return results


@app.get("/get")
async def getTest():
    return {"message": "Get Request Working"}


@app.post("/post")
async def postTest(body: dict):

    return {"message": "Post Request Working", "body": body}
