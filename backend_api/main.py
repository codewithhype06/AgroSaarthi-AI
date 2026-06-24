# ============================================================
# FILE: main.py
# PATH: AgroSaarthi_AI/main.py
# PURPOSE: AgroSaarthi FastAPI Backend - Production Grade
#
# HINGLISH NOTES:
#   - Ye file server ka entry point hai
#   - Startup pe model validate karta hai (size check + magic bytes)
#   - Agar model missing/corrupt hai toh CLEAR error deta hai
#   - /predict-disease endpoint image lekar disease predict karta hai
# ============================================================

import os
import sys
import io
import hashlib
import logging

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

# ─── Logger Setup ───────────────────────────────────────────
# Render ke logs me clearly dikhega kya ho raha hai
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

# ─── Model Path Resolution ───────────────────────────────────
# __file__ se resolve karte hain taaki Render pe bhi sahi kaam kare
# chahe bhi working directory kuch bhi ho
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "agrosaarthi.tflite")


# ─── Startup Diagnostics ────────────────────────────────────
# Ye sab Render build logs me clearly print hoga
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

    # MD5 check - har deploy ke baad compare kar sakte ho
    with open(MODEL_PATH, "rb") as f:
        content = f.read()
        md5 = hashlib.md5(content).hexdigest()
    log.info(f"MD5 Checksum    : {md5}")

    # CRITICAL CHECK: Agar file 100KB se choti hai = Git LFS pointer hai
    # Real .tflite model minimum 1-2 MB hota hai
    if file_size < 100_000:
        log.error("FATAL: Model file too small - likely a Git LFS pointer!")
        log.error("FIX: Run => git add --force agrosaarthi.tflite && git push")
        raise RuntimeError(
            f"Model file is only {file_size} bytes. "
            "This is a Git LFS pointer, not the real model. "
            "Run: git add --force agrosaarthi.tflite"
        )

    # TFLite Magic Bytes check (flatbuffer format validation)
    # Valid .tflite file ke first bytes specific pattern follow karte hain
    if len(content) >= 8:
        # Flatbuffer files have specific identifier bytes
        magic = content[4:8]
        log.info(f"Magic Bytes     : {magic.hex()} (expect tflite flatbuffer)")
    
    log.info("File validation  : PASSED ✓")
else:
    log.error(f"FATAL: Model file NOT FOUND at {MODEL_PATH}")
    log.error("Possible fixes:")
    log.error("  1. Git me file tracked hai? => git ls-files agrosaarthi.tflite")
    log.error("  2. Force add karo => git add --force agrosaarthi.tflite")
    log.error("  3. Render root directory sahi set hai?")
    raise RuntimeError(
        f"TFLite model not found at: {MODEL_PATH}\n"
        "Ensure agrosaarthi.tflite is committed to Git (not in .gitignore). "
        "Run: git add --force agrosaarthi.tflite && git commit && git push"
    )


# ─── TFLite Interpreter Load ────────────────────────────────
# tflite_runtime import: yahi package Render pe use hoga (lightweight)
try:
    import tflite_runtime.interpreter as tflite
    log.info("TFLite Runtime  : tflite_runtime (lightweight) ✓")
except ImportError:
    # Fallback: agar tflite_runtime nahi mila toh tensorflow se try karo
    try:
        import tensorflow.lite as tflite
        log.warning("TFLite Runtime  : tensorflow.lite (fallback) ⚠")
    except ImportError:
        log.error("FATAL: Neither tflite_runtime nor tensorflow is installed!")
        raise RuntimeError(
            "No TFLite runtime found. "
            "Add 'tflite-runtime==2.14.0' to requirements.txt"
        )

try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Input shape verify karo - should be [1, 224, 224, 3]
    input_shape = input_details[0]['shape'].tolist()
    log.info(f"Model Input     : {input_shape} (expect [1, 224, 224, 3])")
    log.info(f"Output Classes  : {output_details[0]['shape'][-1]} (expect 38)")
    log.info("Model Load      : SUCCESS ✓")
    log.info("=" * 55)

except Exception as e:
    log.error(f"FATAL: TFLite model load failed => {e}")
    log.error("Possible causes:")
    log.error("  1. Model file corrupt (re-export from Keras)")
    log.error("  2. tflite_runtime version mismatch")
    log.error("  3. Architecture mismatch (ARM vs x86)")
    raise RuntimeError(f"Failed to load TFLite model: {e}")


# ─── Pydantic Models ────────────────────────────────────────
class LocationData(BaseModel):
    latitude: float
    longitude: float

class ChatRequest(BaseModel):
    disease_name: str
    user_query: str


# ─── Core Prediction Function ────────────────────────────────
def process_image_and_predict(image_bytes: bytes) -> tuple[str, float]:
    """
    Image bytes lekar crop disease predict karta hai.
    
    Steps:
      1. PIL se image open karo
      2. RGB convert + 224x224 resize
      3. [0,1] normalize karo
      4. TFLite run karo
      5. Argmax se class aur confidence nikalo
    """
    try:
        # Step 1: Image open karo (corrupt image handle hoga)
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. JPG, PNG, WEBP supported hai."
            )

        # Step 2: Resize to model input size
        image = image.resize((224, 224), Image.LANCZOS)

        # Step 3: Normalize [0, 255] → [0.0, 1.0]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Step 4: Batch dimension add karo [224,224,3] → [1,224,224,3]
        img_array = np.expand_dims(img_array, axis=0)

        # Step 5: TFLite inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        # Step 6: Output extract karo
        output_data   = interpreter.get_tensor(output_details[0]['index'])
        predicted_idx = int(np.argmax(output_data[0]))
        confidence    = float(np.max(output_data[0])) * 100.0

        # Bounds check (safety)
        if predicted_idx >= len(CLASS_NAMES):
            raise HTTPException(
                status_code=500,
                detail=f"Invalid prediction index: {predicted_idx}"
            )

        return CLASS_NAMES[predicted_idx], confidence

    except HTTPException:
        raise  # Re-raise HTTP errors as-is
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(e)}"
        )


# ─── Risk Level Logic ────────────────────────────────────────
def determine_risk_level(disease_name: str) -> str:
    """Disease name se risk level determine karta hai."""
    if "Healthy" in disease_name:
        return "Low"
    high_risk_keywords = [
        "Blight", "Rust", "Rot", "Virus", "Mold",
        "Spot", "Mildew", "Greening", "Esca"
    ]
    if any(kw in disease_name for kw in high_risk_keywords):
        return "High"
    return "Medium"


# ─── API Endpoints ───────────────────────────────────────────

@app.get("/")
async def root():
    """Health check - Render ka status page ke liye."""
    return {
        "status": "online",
        "service": "AgroSaarthi AI API",
        "version": "2.0.0",
        "model": "TFLite (38 crop disease classes)"
    }


@app.get("/health")
async def health_check():
    """Detailed health check - model loaded hai ya nahi."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "model_size_mb": round(os.path.getsize(MODEL_PATH) / 1e6, 2),
        "classes": len(CLASS_NAMES)
    }


@app.post("/predict-disease")
async def predict_disease(
    file: UploadFile = File(..., description="Crop leaf image (JPG/PNG)"),
    latitude: float  = Form(None, description="GPS latitude (optional)"),
    longitude: float = Form(None, description="GPS longitude (optional)")
):
    """
    Main endpoint: Crop ki photo upload karo, disease predict hogi.
    
    - file: Leaf image (JPG, PNG, WEBP)
    - latitude/longitude: Optional, Firestore logging ke liye
    """
    # File type validate karo
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPG/PNG/WEBP."
        )

    # Image bytes read karo
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    log.info(f"Prediction request: {file.filename} ({len(image_bytes):,} bytes)")

    # Core prediction
    predicted_disease, confidence = process_image_and_predict(image_bytes)
    risk_level = determine_risk_level(predicted_disease)

    log.info(f"Result: {predicted_disease} | Confidence: {confidence:.1f}% | Risk: {risk_level}")

    # Optional: Firebase logging
    db_response = None
    if latitude is not None and longitude is not None:
        try:
            from app.firebase_db import save_disease_outbreak
            db_response = save_disease_outbreak(
                predicted_disease, latitude, longitude, risk_level
            )
        except Exception as e:
            log.warning(f"Firebase logging failed (non-critical): {e}")
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
    """Location ke basis pe weather-based disease risk calculate karo."""
    try:
        from app.weather import get_weather_risk
        result = get_weather_risk(location.latitude, location.longitude)
        return {"status": "success", "weather_analysis": result}
    except Exception as e:
        log.error(f"Weather API error: {e}")
        raise HTTPException(status_code=503, detail=f"Weather service unavailable: {e}")


@app.post("/ask-agronomist")
async def ask_agronomist(chat_data: ChatRequest):
    """RAG chatbot: Disease ke baare mein agronomist se poocho."""
    try:
        from app.rag_chatbot import get_agronomist_response
        response = get_agronomist_response(chat_data.disease_name, chat_data.user_query)
        return {"status": "success", "bot_response": response}
    except Exception as e:
        log.error(f"Chatbot error: {e}")
        raise HTTPException(status_code=503, detail=f"Chatbot unavailable: {e}")