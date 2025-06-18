from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# Configure CORS middleware
# This tells the browser that requests from any origin (*) are allowed
# to access backend. For production, you might want to replace "*"
# with the specific URL of the frontend application.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the TensorFlow model
MODEL = tf.keras.models.load_model("./potatoes.h5")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# server-check endpoint
@app.get("/ping")
async def ping():
    return "Potato Disease Classification Server!"

# Function to convert uploaded file bytes to a NumPy array image
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = np.array(image)
    return image

# predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image file
    image = read_file_as_image(await file.read())

    # Prepare image for model prediction (add batch dimension)
    image_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(image_batch)

    # Get the predicted class and confidence
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    print("predicted_class = ", predicted_class)
    print("confidence = ", confidence)

    # Return the prediction and confidence as JSON
    # IMPORTANT: Ensure the keys here match what your frontend expects
    return {
        'prediction': predicted_class, # Changed 'class' to 'prediction'
        'accuracy': float(confidence)  # Changed 'confidence' to 'accuracy' and ensured float type
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)