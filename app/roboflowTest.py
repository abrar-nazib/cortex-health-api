from roboflow import Roboflow

rf = Roboflow(api_key="Wkb7m6Ft10UqTymiTlLr")
project = rf.workspace().project("brain-cancer-detection-mri-images")
model = project.version(2).model

# infer on a local image
print(model.predict("../temp/temp.jpg").json())

# # infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True).json())

# # save an image annotated with your predictions
# model.predict("your_image.jpg").save("prediction.jpg")
