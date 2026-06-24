# FILE: rag_chatbot.py
# PATH: AgroSaarthi_AI/backend_api/app/rag_chatbot.py
# PURPOSE: Direct REST API connection to Gemini (Bypassing SDK errors)

import os
import requests

# Fetch the secret API key from Render Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def get_agronomist_response(disease_name: str, user_query: str) -> str:
    if not GEMINI_API_KEY:
        return "System Warning: Render par GEMINI_API_KEY environment variable set nahi hai."

    # Step 1: Prepare the AI Persona and Prompt
    prompt = f"""
    You are 'AgroSaarthi', an expert AI Agronomist working in India.
    A farmer is asking about the following crop disease: '{disease_name}'
    Their specific question is: '{user_query}'
    
    Instructions:
    1. Provide a step-by-step treatment plan.
    2. Include both organic and chemical methods if applicable.
    3. The language MUST be simple Hinglish (Hindi written in English alphabet), easily understandable by an Indian farmer.
    4. Keep it friendly and professional. Start with "Namaste Kisan Bhai!".
    """
    
    # Step 2: Direct REST API URL for Gemini 1.5 Flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    # Step 3: Prepare the exact JSON structure Google expects
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        # Step 4: Make the direct network request
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        
        # Step 5: Check if successful
        if response.status_code == 200:
            data = response.json()
            # Extracting the text from Google's JSON format
            bot_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return bot_text.strip()
        else:
            # If Google rejects the key or model, this will print the exact reason
            return f"DEBUG ERROR (API REST): Code {response.status_code} -> {response.text}"
            
    except Exception as e:
        return f"DEBUG ERROR (Network): {str(e)}"