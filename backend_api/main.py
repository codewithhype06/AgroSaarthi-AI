import os
import sys
import io
import hashlib
import logging
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

# ─── TFLite Import (Render compatible) ──────────────────────
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ─── Logger Setup ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("agrosaarthi")

# ─── App Init ───────────────────────────────────────────────
app = FastAPI(
    title="AgroSaarthi AI API",
    description="Crop Disease Detection via TFLite - VisionX Backend",
    version="2.0.0"
)

# ─── Class Labels (38 classes) ──────────────────────────────
CLASS_NAMES = [
    'Apple - Scab', 'Apple - Black Rot', 'Apple - Cedar Rust', 'Apple - Healthy',
    'Blueberry - Healthy',
    'Cherry - Powdery Mildew', 'Cherry - Healthy',
    'Corn - Cercospora Leaf Spot', 'Corn - Common Rust',
    'Corn - Northern Leaf Blight', 'Corn - Healthy',
    'Grape - Black Rot', 'Grape - Esca', 'Grape - Leaf Blight', 'Grape - Healthy',
    'Orange - Citrus Greening',
    'Peach - Bacterial Spot', 'Peach - Healthy',
    'Pepper - Bacterial Spot', 'Pepper - Healthy',
    'Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy',
    'Raspberry - Healthy',
    'Soybean - Healthy',
    'Squash - Powdery Mildew',
    'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight',
    'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites',
    'Tomato - Target Spot', 'Tomato - Yellow Leaf Curl Virus',
    'Tomato - Mosaic Virus', 'Tomato - Healthy'
]

# ─── Model Path ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "agrosaarthi.tflite")

# ─── Startup Diagnostics ────────────────────────────────────
log.info("=" * 55)
log.info("AGROSAARTHI STARTUP DIAGNOSTICS")
log.info("=" * 55)
log.info(f"Python Version  : {sys.version.split()[0]}")
log.info(f"BASE_DIR        : {BASE_DIR}")
log.info(f"MODEL_PATH      : {MODEL_PATH}")
log.info(f"File Exists     : {os.path.exists(MODEL_PATH)}")

if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH)
    log.info(f"File Size       : {file_size:,} bytes ({file_size / 1e6:.2f} MB)")
    with open(MODEL_PATH, "rb") as f:
        content = f.read()
    log.info(f"MD5 Checksum    : {hashlib.md5(content).hexdigest()}")
    if file_size < 100_000:
        raise RuntimeError(
            f"Model file is only {file_size} bytes - Git LFS pointer hai. "
            "Run: git add --force agrosaarthi.tflite && git push"
        )
    log.info("File validation  : PASSED ✓")
else:
    raise RuntimeError(
        f"Model NOT FOUND at {MODEL_PATH}. "
        "Run: git add --force agrosaarthi.tflite && git commit && git push"
    )

# ─── Load TFLite Model ──────────────────────────────────────
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    log.info(f"Model Input     : {input_details[0]['shape'].tolist()}")
    log.info(f"Output Classes  : {output_details[0]['shape'][-1]}")
    log.info("Model Load      : SUCCESS ✓")
    log.info("=" * 55)
except Exception as e:
    raise RuntimeError(f"Failed to load TFLite model: {e}")

# ─── Pydantic Models ────────────────────────────────────────
class LocationData(BaseModel):
    latitude: float
    longitude: float

class ChatRequest(BaseModel):
    disease_name: str
    user_query: str

# ─── Core Prediction ────────────────────────────────────────
def process_image_and_predict(image_bytes: bytes) -> tuple[str, float]:
    try:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        image     = image.resize((224, 224), Image.LANCZOS)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        output_data   = interpreter.get_tensor(output_details[0]['index'])
        predicted_idx = int(np.argmax(output_data[0]))
        confidence    = float(np.max(output_data[0])) * 100.0

        if predicted_idx >= len(CLASS_NAMES):
            raise HTTPException(status_code=500, detail=f"Invalid index: {predicted_idx}")

        return CLASS_NAMES[predicted_idx], confidence

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

# ─── Risk Level ─────────────────────────────────────────────
def determine_risk_level(disease_name: str) -> str:
    if "Healthy" in disease_name:
        return "Low"
    if any(kw in disease_name for kw in ["Blight","Rust","Rot","Virus","Mold","Spot","Mildew","Greening","Esca"]):
        return "High"
    return "Medium"

# ─── Endpoints ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "online", "service": "AgroSaarthi AI API", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_size_mb": round(os.path.getsize(MODEL_PATH) / 1e6, 2),
        "classes": len(CLASS_NAMES)
    }

@app.post("/predict-disease")
async def predict_disease(
    file: UploadFile = File(...),
    latitude: float  = Form(None),
    longitude: float = Form(None)
):
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file received.")

    log.info(f"Prediction request: {file.filename} ({len(image_bytes):,} bytes)")

    predicted_disease, confidence = process_image_and_predict(image_bytes)
    risk_level = determine_risk_level(predicted_disease)

    log.info(f"Result: {predicted_disease} | {confidence:.1f}% | Risk: {risk_level}")

    db_response = None
    if latitude is not None and longitude is not None:
        try:
            from app.firebase_db import save_disease_outbreak
            db_response = save_disease_outbreak(predicted_disease, latitude, longitude, risk_level)
        except Exception as e:
            log.warning(f"Firebase logging failed: {e}")
            db_response = {"warning": "DB log failed but prediction succeeded"}

    return {
        "status": "success",
        "filename": file.filename,
        "prediction": predicted_disease,
        "confidence_score": f"{confidence:.1f}%",
        "confidence_raw": round(confidence, 2),
        "risk_level": risk_level,
        "database_log": db_response,
        "message": "Prediction via local TFLite engine."
    }

@app.post("/weather-risk")
async def calculate_weather_risk(location: LocationData):
    try:
        from app.weather import get_weather_risk
        result = get_weather_risk(location.latitude, location.longitude)
        return {"status": "success", "weather_analysis": result}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Weather service unavailable: {e}")

@app.post("/ask-agronomist")
async def ask_agronomist(chat_data: ChatRequest):
    try:
        from app.rag_chatbot import get_agronomist_response
        response = get_agronomist_response(chat_data.disease_name, chat_data.user_query)
        return {"status": "success", "bot_response": response}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Chatbot unavailable: {e}")