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
import psycopg2
from psycopg2.extras import RealDictCursor
import time
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


while True:
    try:
        connection = psycopg2.connect(user="postgres",
                                      password="robot",
                                      host="localhost",
                                      port="5432",
                                      database="cortex",
                                      cursor_factory=RealDictCursor)
        # cursor_factory is used to return the result as a dictionary because normally column names are not returned
        cursor = connection.cursor()  # create a cursor instance
        # print a colored text to terminal
        print("\033[92m" + "INFO:     Database connected successfully" + "\033[0m")

        break  # break the loop if connection is successful
    except Exception as e:
        print("Connection to database failed")
        print(f"Error: {e}")
        time.sleep(2)  # give some time so that the database can start
        exit(1)  # exit the program if connection fails


def create_classifier_dict(class_name, confidence):
    results = {
        "predicted-class": f"{class_name}",
        "confidence": float(f"{confidence}"),
    }
    return results


def create_box_dict(class_name, confidence, x, y, w, h, img_w=512, img_h=512):
    # convert every value to float
    confidence, x, y, w, h, img_w, img_h = float(confidence), float(x), float(y), float(w), float(h), float(
        img_w), float(img_h)
    # calculate the center of the box
    cx, cy = x + (w / 2), y + (h / 2)
    # calculate the radius of the box
    radius = w/2 if w < h else h/2

    # scale the values according to the image size
    cx, cy = (cx * (300/img_w))-150 - \
        (radius/2), (cy * (300/img_h))-150 - (radius/2)
    radius = radius * (300/img_w) + 2
    elem = {
        "predicted-class": f"{class_name}",
        "confidence": confidence,
        "cx": int(cx),
        "cy": int(cy),
        "radius": int(radius),
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
    results["box"] = {}
    print(results)
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

    locate_liver_disease = liver_model.predict(f"{project_root}/temp/temp.jpg",
                                               hosted=False, confidence=40, overlap=30).json()
    predictions = locate_liver_disease["predictions"]
    img_w, img_h = locate_liver_disease["image"]["width"], locate_liver_disease["image"]["height"]
    return_dict = {}

    if (predictions != []):
        # sort the predictions by confidence score
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        # get the top prediction
        top_prediction = predictions[0]
        box_dict = create_box_dict(class_name=top_prediction["class"],
                                   confidence=top_prediction["confidence"],
                                   x=top_prediction["x"],
                                   y=top_prediction["y"],
                                   w=top_prediction["width"],
                                   h=top_prediction["height"],
                                   img_w=img_w,
                                   img_h=img_h)
        return_dict = create_classifier_dict(
            top_prediction["class"], top_prediction["confidence"])
        return_dict["box"] = box_dict
    else:
        return_dict = create_classifier_dict("No Liver Disease", 1)
        return_dict["box"] = {}
    print(return_dict)
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

    try:
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
        img_w, img_h = locateTumor["image"]["width"], locateTumor["image"]["height"]

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
                                                     top_prediction["height"],
                                                     img_w=img_w,
                                                     img_h=img_h,
                                                     )

        print(classifier_dict)
    except Exception as e:
        print(e)
        return {"message": "Error Occured"}
    return classifier_dict


@app.post("/api/data-collector/collect/hematology")
async def collect_data(body: dict):
    data = {
        "white_blood_cells": 8.30,
        "neutrophil": 4.57,
        "lymphocyte": 3.24,
        "monocyte": 0.33,
        "esinophil": 0.17,
        "basophil": 0.00,
        "red_blood_cells": 5.30,
        "hemoglobin": 14.80,
        "hct": 45.20,
        "mcv": 85.30,
        "mch": 27.90,
        "mchc": 32.80,
        "plateles": 347.00,
        "mpv": 9.30,
        "pct": 0.32,
        "pdw": 9.90,
        "reticulocyte": 0.00,
        "esr": 90,
        "label": "typhoid"
    }
    # create insert SQL query from data dictionary
    query = "INSERT INTO hematology ("
    for key in data.keys():
        query += f"{key}, "
    query = query[:-2] + ") VALUES ("
    for key in data.keys():
        # put sanitized values in the query
        if (key == "label"):
            query += f"'{data[key]}', "
        else:
            query += f"{data[key]}, "

    query = query[:-2] + ") RETURNING *;"
    # print(query)
    cursor.execute(query)  # for sanitization, use %s instead of {}
    data = cursor.fetchone()
    connection.commit()
    cursor.close()
    connection.close()
    # check whether the data was saved successfully
    return data


@app.get("/get")
async def getTest():
    return {"message": "Get Request Working"}


@app.post("/post")
async def postTest(body: dict):

    return {"message": "Post Request Working", "body": body}
