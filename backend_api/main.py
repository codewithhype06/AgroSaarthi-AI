# FILE: main.py
# PATH: AgroSaarthi_AI/backend_api/main.py
# PURPOSE: Production-Grade TFLite Execution (Strict, Error-Free, Crash-Free)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import io
import os
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from app.weather import get_weather_risk
from app.rag_chatbot import get_agronomist_response
from app.firebase_db import save_disease_outbreak

app = FastAPI(
    title="AgroSaarthi AI API", 
    description="VisionX Production Backend",
    version="1.0.0"
)

class LocationData(BaseModel):
    latitude: float
    longitude: float

class ChatRequest(BaseModel):
    disease_name: str
    user_query: str

CLASS_NAMES = ['Apple - Scab', 'Apple - Black Rot', 'Apple - Cedar Rust', 'Apple - Healthy', 'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy', 'Corn - Cercospora Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy', 'Grape - Black Rot', 'Grape - Esca', 'Grape - Leaf Blight', 'Grape - Healthy', 'Orange - Citrus Greening', 'Peach - Bacterial Spot', 'Peach - Healthy', 'Pepper - Bacterial Spot', 'Pepper - Healthy', 'Potato - Early Blight', 'Potato Late Blight', 'Potato - Healthy', 'Raspberry - Healthy', 'Soybean - Healthy', 'Squash - Powdery Mildew', 'Strawberry - Leaf Scorch', 'Strawberry - Healthy', 'Tomato - Bacterial Spot', 'Tomato Early Blight', 'Tomato - Late Blight', 'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites', 'Tomato - Target Spot', 'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy']

# --- STRICT STARTUP INITIALIZATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "agrosaarthi.tflite")

# Strict check: Server start hone se pehle model file confirm karega
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"CRITICAL SYSTEM ERROR: TFLite model file missing at {MODEL_PATH}. Check your Git upload!")

# Load model to memory at startup (Production Best Practice)
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    raise RuntimeError(f"CRITICAL SYSTEM ERROR: Failed to load TFLite model. Details: {str(e)}")

def process_image_and_predict(image_bytes: bytes) -> tuple[str, float]:
    try:
        # 1. Standardize Image Input
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 2. Execute TFLite Model Local Engine
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        # 3. Extract Prediction and Confidence
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_idx = int(np.argmax(output_data[0]))
        confidence = float(np.max(output_data[0])) * 100.0
        
        return CLASS_NAMES[predicted_idx], confidence
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

# --- ENDPOINTS ---

@app.post("/predict-disease")
async def predict_disease(
    file: UploadFile = File(...),
    latitude: float = Form(None),
    longitude: float = Form(None)
):
    image_bytes = await file.read()
    
    # Core AI Logic
    predicted_disease, confidence = process_image_and_predict(image_bytes)
    
    # Risk Level Assessment
    if "Healthy" in predicted_disease:
        risk_level = "Low"
    elif any(keyword in predicted_disease for keyword in ["Blight", "Rust", "Spot", "Mold", "Rot", "Virus"]):
        risk_level = "High"
    else:
        risk_level = "Medium"
    
    # Database Logging
    db_response = None
    if latitude is not None and longitude is not None:
        db_response = save_disease_outbreak(predicted_disease, latitude, longitude, risk_level)
    
    return {
        "status": "success",
        "filename": file.filename,
        "prediction": predicted_disease,
        "confidence_score": f"{confidence:.1f}%", 
        "database_log": db_response,
        "message": "Real prediction successful via Local TFLite Engine."
    }

@app.get("/")
async def root(): 
    return {"status": "success", "message": "VisionX AI Backend Online"}

@app.get("/health")
async def health_check(): 
    return {"status": "online", "model_loaded": True}

@app.post("/weather-risk")
async def calculate_weather_risk(location: LocationData):
    return {"status": "success", "weather_analysis": get_weather_risk(location.latitude, location.longitude)}

@app.post("/ask-agronomist")
async def ask_agronomist(chat_data: ChatRequest):
    return {"status": "success", "bot_response": get_agronomist_response(chat_data.disease_name, chat_data.user_query)}