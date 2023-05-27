import segno
import json


def generateQRCode(data):
    qr = segno.make(data)
    qr.save('qrCode.png', scale=10)
    return True


# # create dictionary
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
}


# convert into JSON:
json_data = json.dumps(data)

generateQRCode(json_data)
