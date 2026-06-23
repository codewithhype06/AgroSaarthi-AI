# FILE: main.py
# PATH: AgroSaarthi_AI/backend_api/main.py
# PURPOSE: Entry point for AgroSaarthi AI Backend Server

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import asyncio
from app.weather import get_weather_risk
from app.rag_chatbot import get_agronomist_response
from app.firebase_db import save_disease_outbreak

# Initialize FastAPI application
app = FastAPI(
    title="AgroSaarthi AI API",
    description="Backend server for crop disease detection and RAG Chatbot",
    version="1.0.0"
)

class LocationData(BaseModel):
    latitude: float
    longitude: float

class ChatRequest(BaseModel):
    disease_name: str
    user_query: str

@app.get("/")
async def root():
    return {"status": "success", "message": "Welcome to AgroSaarthi AI Backend!"}

@app.get("/health")
async def health_check():
    return {"status": "online", "service": "AgroSaarthi API"}

# UPDATED: Predict Disease Endpoint
@app.post("/predict-disease")
async def predict_disease(
    file: UploadFile = File(...),
    latitude: float = Form(None),   # Form se data lene ke liye
    longitude: float = Form(None)
):
    filename = file.filename
    await asyncio.sleep(2) # Dummy delay
    
    predicted_disease = "Potato Late Blight"
    risk_level = "High"
    
    # DB Save Logic (Only if location is provided)
    db_response = None
    if latitude is not None and longitude is not None:
        db_response = save_disease_outbreak(predicted_disease, latitude, longitude, risk_level)
    
    return {
        "status": "success",
        "filename": filename,
        "prediction": predicted_disease,
        "confidence_score": "94.5%",
        "database_log": db_response,
        "message": "Prediction successful and logged to DB."
    }

@app.post("/weather-risk")
async def calculate_weather_risk(location: LocationData):
    risk_data = get_weather_risk(location.latitude, location.longitude)
    return {"status": "success", "weather_analysis": risk_data}

@app.post("/ask-agronomist")
async def ask_agronomist(chat_data: ChatRequest):
    ai_response = get_agronomist_response(chat_data.disease_name, chat_data.user_query)
    return {"status": "success", "bot_response": ai_response}