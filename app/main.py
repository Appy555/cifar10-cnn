from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf
import numpy as np
import os


app = FastAPI(
    title="CIFAR-10 CNN API",
    description="CIFAR-10 image classification using CNN",
    version="1.0.0"
)


# CIFAR-10 classes
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cifar10_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "CIFAR-10 CNN API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "cifar10_model.keras"
    }


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_data = await file.read()

    # Convert image bytes to PIL image
    image = Image.open(
        __import__("io").BytesIO(image_data)
    )

    # Convert to RGB
    image = image.convert("RGB")

    # CIFAR-10 input size
    image = image.resize((32, 32))

    # Convert to NumPy
    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Highest probability
    predicted_index = int(
        np.argmax(predictions[0])
    )

    predicted_class = classes[
        predicted_index
    ]

    confidence = float(
        predictions[0][predicted_index]
    )

    return {
        "predicted_class": predicted_class,
        "confidence": round(
            confidence * 100,
            2
        )
    }