# FILE: firebase_db.py
# PATH: AgroSaarthi_AI/backend_api/app/firebase_db.py
# PURPOSE: Firebase Firestore connection and data logging

import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import os

# Check if the real key file exists (we will add this in Phase 11 Deployment)
KEY_FILE_PATH = "serviceAccountKey.json"

if os.path.exists(KEY_FILE_PATH):
    cred = credentials.Certificate(KEY_FILE_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    USE_MOCK_DB = False
else:
    db = None
    USE_MOCK_DB = True
    print("⚠️ WARNING: serviceAccountKey.json not found. Using MOCK Database mode for development.")

def save_disease_outbreak(disease_name: str, lat: float, lon: float, risk_level: str):
    """
    Saves the detection event to Firebase. If no key is present, prints to console.
    """
    record = {
        "disease": disease_name,
        "latitude": lat,
        "longitude": lon,
        "risk_level": risk_level,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    
    if USE_MOCK_DB:
        # Dummy save for local development
        print(f"🔥 [MOCK DB SAVE] New Outbreak Logged -> {record}")
        return {"status": "mock_saved", "record_id": "dummy_12345"}
    else:
        # Real save to Google Cloud Firestore
        doc_ref = db.collection(u'disease_outbreaks').document()
        doc_ref.set(record)
        return {"status": "saved_to_firebase", "record_id": doc_ref.id}