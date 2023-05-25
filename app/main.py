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
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import imghdr
from roboflow import Roboflow
# import uvicorn
# from uvicorn.config import LOGGING_CONFIG
# LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"

# fmt: on

# get the path of the root of the project directory
project_root = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(project_root)


# get the absolute path of the directory where this file is located
dir_path = os.path.dirname(os.path.realpath(__file__))
model_dir_path = os.path.join(dir_path, "../models")

app = FastAPI()


np.set_printoptions(suppress=True)  # suppress scientific notations

# Load Models
rf = Roboflow(api_key="Wkb7m6Ft10UqTymiTlLr")

pneumoniaDetectorModel = load_model(
    f"{model_dir_path}/pneumonia-predictor/pneumonia-predictor.h5")
pneumoniaClassNames = [line.rstrip('\n') for line in open(
    f"{model_dir_path}/pneumonia-predictor/labels.txt", "r")]

# Load Model: Liver Disease predictor
liver_project = rf.workspace().project("liver-disease")
liver_model = liver_project.version(1).model

# Load model: Brain Tumor predictor
brain_tumor_project = rf.workspace().project("tumor-detection-j9mqs")
brain_tumor_model = brain_tumor_project.version(1).model

# Load model: Brain Tumor predictor Binary
brain_tumor_binary_project = rf.workspace().project(
    "brain-cancer-detection-mri-images")
brain_tumor_binary_model = brain_tumor_binary_project.version(2).model


def create_classifier_dict(class_name, confidence):
    results = {
        "predicted-class": f"{class_name}",
        "confidence": f"{confidence}",
    }
    return results


def create_box_dict(class_name, confidence, x, y, w, h):
    cx, cy = int(x + w/2), int(y + h/2)
    radius = int(w/2) if w < h else int(h/2)
    elem = {
        "predicted-class": f"{class_name}",
        "confidence": f"{confidence}",
        "cx": f"{cx}",
        "cy": f"{cy}",
        "radius": f"{radius}",
    }
    return elem


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
    result = create_classifier_dict(class_name, confidence_score)
    return result


def validateImage(file):
    valid_image_types = ["jpg", "jpeg", "png", "gif"]
    file_type = imghdr.what(file.file)
    # check if the file is an image
    if file_type not in valid_image_types:
        return False
    return True


@app.post("/api/models/pneumonia-predictor/predict")
async def predict_pneumonia(file: UploadFile = File(...)):
    # check if the file is an image
    if not validateImage(file):
        raise HTTPException(
            status_code=400, detail="Only image files are allowed.")

    # read the image file using pillow
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    results = inference_pneumonia_predictor(image)
    return results


@app.post("/api/models/liver-disease-predictor/predict")
async def predict_liver_disease(file: UploadFile = File(...)):
    # check if the file is an image
    if not validateImage(file):
        raise HTTPException(
            status_code=400, detail="Only image files are allowed.")
    # read the image file using pillow
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    # save the image to a temporary directory
    temp_image_path = f"{project_root}/temp/temp.jpg"
    image.save(temp_image_path)

    result = liver_model.predict(f"{project_root}/temp/temp.jpg",
                                 hosted=False, confidence=40, overlap=30).json()
    predictions = result["predictions"]
    return_dict = {}

    if (predictions != []):
        # sort the predictions by confidence score
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        # get the top prediction
        top_prediction = predictions[0]
        box_dict = create_box_dict(top_prediction["class"],
                                   top_prediction["confidence"],
                                   top_prediction["x"],
                                   top_prediction["y"],
                                   top_prediction["width"],
                                   top_prediction["height"])
        return_dict = create_classifier_dict(
            top_prediction["class"], top_prediction["confidence"])
        return_dict["box"] = box_dict
    else:
        return_dict = create_classifier_dict("No Liver Disease", 1)
        return_dict["box"] = {}
    return return_dict


@app.post("/api/models/brain-tumor-predictor/predict")
async def predict_brain_tumor(file: UploadFile = File(...)):
    # chek if the file is an image
    if not validateImage(file):
        raise HTTPException(
            status_code=400, detail="Only image files are allowed.")
    # read the image file using pillow
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    # save the image to a temporary directory
    temp_image_path = f"{project_root}/temp/temp.jpg"
    image.save(temp_image_path)

    # Check if there is a tumor
    checkTumor = brain_tumor_binary_model.predict(temp_image_path).json()
    predictions = checkTumor["predictions"][0]
    predicted_class = predictions["predicted_classes"][0]
    prediction_object = predictions["predictions"][predicted_class]
    confidence = prediction_object["confidence"]
    if predicted_class == "notumor":
        predicted_class = "Not Detected"
    classifier_dict = create_classifier_dict(predicted_class, confidence)
    classifier_dict["box"] = {}

    locateTumor = brain_tumor_model.predict(f"{project_root}/temp/temp.jpg",
                                            hosted=False, confidence=40, overlap=30).json()
    # sort the predictions by confidence score
    if (locateTumor["predictions"] != []):
        locateTumor["predictions"].sort(
            key=lambda x: x["confidence"], reverse=True)
        # get the top prediction
        top_prediction = locateTumor["predictions"][0]
        classifier_dict["box"] = create_box_dict("tumor",
                                                 top_prediction["confidence"],
                                                 top_prediction["x"],
                                                 top_prediction["y"],
                                                 top_prediction["width"],
                                                 top_prediction["height"])

    print(classifier_dict)
    return classifier_dict


@app.get("/get")
async def getTest():
    return {"message": "Get Request Working"}


@app.post("/post")
async def postTest(body: dict):

    return {"message": "Post Request Working", "body": body}

# if __name__ == "__main__":
#     uvicorn.run(app, host="192.168.43.113", port=80)
