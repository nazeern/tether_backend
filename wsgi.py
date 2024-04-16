from dataclasses import asdict
from dataclasses import dataclass
from flask import Flask
from flask import request

import os
import pickle


from transformers import Flatten, Grayscaler, Resizer

from sklearn.pipeline import Pipeline

from transformers import Flatten, Grayscaler, Resizer
from validators import ImageValidator

app = Flask(__name__)
print(app.root_path)

@dataclass
class PredictImageResponse:
    response: int | None
    error: str | None


@app.get("/")
def get_app_status():
    return "Success!"


@app.post("/predict")
def predict_image() -> PredictImageResponse:
    """
    Expects: an input image provided in the request.

    Returns: either a response or an error message.
    """

    # Check for image in request data
    filestorage = request.files.get('image')

    # Validate image dimensions are within bounds
    validator = ImageValidator(xmax=3000, ymax=5000)
    arr, error = validator.validate(filestorage)

    print(error)

    if error:
        return asdict(PredictImageResponse(
            response=None,
            error=error
        )), 400

    # Transform image to prepare for ML model
    transformers = (
        Resizer(height=50, width=89),
        Grayscaler(),
        Flatten()
    )
    for tf in transformers:
        arr = tf.transform(arr)

    # Input prepared image into ML model
    with open('pipeline.pkl', 'rb') as file:
        pipeline: Pipeline = pickle.load(file)
    
    pred = pipeline.predict(arr.reshape(1, -1))[0]

    # Return a response
    return asdict(PredictImageResponse(
        response=int(pred),
        error=None,
    )), 200