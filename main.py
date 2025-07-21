import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx # For making asynchronous HTTP requests
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Gemini Chatbot & Potato Disease Classification API",
    description="A FastAPI application combining a Gemini chatbot with a potato disease classification model.",
    version="1.0.0"
)

# Configure CORS middleware
# This tells the browser that requests from any origin (*) are allowed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Gemini Chatbot Specifics ---
# Pydantic model for the chatbot request body
class ChatRequest(BaseModel):
    message: str

# Retrieve Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL")

@app.post("/chat")
async def chat_with_gemini(request: ChatRequest):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        raise HTTPException(
            status_code=500,
            detail="Gemini API Key is not configured. Please set GEMINI_API_KEY in your .env file."
        )

    # --- Define a system instruction to guide the Gemini model ---
    system_instruction = (
        "You are an expert assistant specializing in potato and potato leaf diseases and potato disease classification."
        "Provide detailed and accurate information about potato diseases, "
        "their symptoms, causes, prevention methods, and treatment options. "
        "If a question is outside the scope of potato leaf diseases, kindly state that "
        "you can only answer questions related to potato leaf health."
    )

    # Combine the system instruction with the user's message
    # We send both as part of the conversation history.
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": system_instruction + "\n\nUser query: " + request.message}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7, # Controls the randomness of the output. Higher values mean more random.
            "maxOutputTokens": 500 # Increased max tokens for potentially detailed disease info
        }
    }
    # --- End of Modification ---

    headers = {
        "Content-Type": "application/json"
    }

    # Make the asynchronous API call to Gemini
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            gemini_response = response.json()

            # Extract the text from the Gemini response
            if gemini_response and "candidates" in gemini_response and \
               len(gemini_response["candidates"]) > 0 and \
               "content" in gemini_response["candidates"][0] and \
               "parts" in gemini_response["candidates"][0]["content"] and \
               len(gemini_response["candidates"][0]["content"]["parts"]) > 0:
                bot_message = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
            else:
                bot_message = "I'm sorry, I couldn't generate a response related to potato leaf diseases."
                print(f"Unexpected Gemini response structure: {gemini_response}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to parse Gemini API response."
                )

            return {"response": bot_message}

        except httpx.RequestError as exc:
            # Handle network errors (e.g., DNS resolution failed, connection refused)
            print(f"An error occurred while requesting Gemini API: {exc}")
            raise HTTPException(
                status_code=503,
                detail=f"Service unavailable: Could not connect to Gemini API."
            )
        except httpx.HTTPStatusError as exc:
            # Handle HTTP errors (e.g., 400, 401, 404, 500 from Gemini API)
            print(f"Gemini API returned an error: {exc.response.status_code} - {exc.response.text}")
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Gemini API error: {exc.response.text}"
            )
        except Exception as exc:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred: {exc}")
            raise HTTPException(
                status_code=500,
                detail="An internal server error occurred."
            )

# --- Potato Disease Classification Specifics ---
try:
    MODEL = tf.keras.models.load_model("./potatoes.keras")
    print("TensorFlow model 'potatoes.h5' loaded successfully.")
except Exception as e:
    print(f"Error loading TensorFlow model: {e}")
    MODEL = None # Set to None if loading fails, and handle in endpoint

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Function to convert uploaded file bytes to a NumPy array image
def read_file_as_image(data) -> np.ndarray:
    """
    Reads image bytes, opens with PIL, converts to NumPy array.
    Ensures the image is in RGB format.
    """
    image = Image.open(BytesIO(data))
    # Ensure the image is in RGB format if it's grayscale or has an alpha channel
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # If your model expects a specific size, you might need to resize here, e.g.:
    # image = image.resize((256, 256)) # Assuming your model expects 256x256
    image = np.array(image)
    return image

# predict-potato-leaf-image-endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check server logs for model loading errors."
        )
    
    # Read the uploaded image file
    image = read_file_as_image(await file.read())

    # Prepare image for model prediction (add batch dimension)
    # This assumes your model expects an input shape like (batch_size, height, width, channels).
    # If your model requires specific preprocessing (e.g., normalization, resizing),
    image_batch = np.expand_dims(image, 0)
    try:
        predictions = MODEL.predict(image_batch)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during model prediction: {e}. Ensure image shape matches model input."
        )

    # Get the predicted class and confidence
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    print("predicted_class = ", predicted_class)
    print("confidence = ", confidence)

    # Return the prediction and confidence as JSON
    # IMPORTANT: Ensure the keys here match what your frontend expects
    return {
        'prediction': predicted_class,
        'accuracy': float(confidence)
    }

# server-check endpoint
@app.get("/ping")
async def ping():
    model_status = "loaded" if MODEL else "not loaded (check console for errors)"
    return {"message": "Potato Disease Classification Server!", "model_status": model_status}

if __name__ == "__main__":
    # Access environment variables for host and port
    host = os.getenv("APP_HOST", "localhost") # Default to 'localhost' if not found
    port = int(os.getenv("APP_PORT", 8000)) # Default to 8000, convert to int

    uvicorn.run(app, host=host, port=port)