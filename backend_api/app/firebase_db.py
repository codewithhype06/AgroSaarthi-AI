# FILE: firebase_db.py
# PATH: AgroSaarthi_AI/backend_api/app/firebase_db.py
# PURPOSE: Firebase Firestore connection securely using Environment Variables

import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import os
import json

# Fetching the secret key from Render Environment Variables
firebase_env = os.environ.get("FIREBASE_CREDENTIALS")

if firebase_env:
    try:
        # Convert string back to JSON dictionary
        cred_dict = json.loads(firebase_env)
        cred = credentials.Certificate(cred_dict)
        
        # Check if already initialized to prevent crashes
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            
        db = firestore.client()
        USE_MOCK_DB = False
        print("✅ SUCCESS: Connected to REAL Firebase Cloud Firestore!")
    except Exception as e:
        db = None
        USE_MOCK_DB = True
        print(f"❌ ERROR: Failed to parse Firebase Credentials: {e}")
else:
    db = None
    USE_MOCK_DB = True
    print("⚠️ WARNING: FIREBASE_CREDENTIALS env var not found. Using MOCK Database.")

def save_disease_outbreak(disease_name: str, lat: float, lon: float, risk_level: str):
    """
    Saves the detection event to Firebase or Console depending on auth status.
    """
    record = {
        "disease": disease_name,
        "latitude": lat,
        "longitude": lon,
        "risk_level": risk_level,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    
    if USE_MOCK_DB:
        print(f"🔥 [MOCK DB SAVE] -> {record}")
        return {"status": "mock_saved", "record_id": "dummy_123"}
    else:
        # Real save to Google Cloud Firestore
        doc_ref = db.collection(u'disease_outbreaks').document()
        doc_ref.set(record)
        return {"status": "saved_to_firebase", "record_id": doc_ref.id}