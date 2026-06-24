# FILE: rag_chatbot.py
# PATH: AgroSaarthi_AI/backend_api/app/rag_chatbot.py
# PURPOSE: GenAI Chatbot Logic (DEBUG MODE)

import os
import google.generativeai as genai

# Fetch the secret API key from Render Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    USE_MOCK_AI = False
else:
    USE_MOCK_AI = True
    print("⚠️ WARNING: GEMINI_API_KEY not found.")

def get_agronomist_response(disease_name: str, user_query: str) -> str:
    if USE_MOCK_AI:
        return "System Warning: Render par GEMINI_API_KEY environment variable set nahi hai ya khali hai."

    prompt = f"""
    You are 'AgroSaarthi', an expert AI Agronomist working in India.
    A farmer is asking about the following crop disease: '{disease_name}'
    Their specific question is: '{user_query}'
    
    Instructions: Provide a step-by-step treatment plan in simple Hinglish. Start with "Namaste Kisan Bhai!".
    """
    
    try:
        response = model.generate_content(prompt)
        if response.text:
            return response.text.strip()
        else:
            return "Error: AI ne empty response diya."
            
    except Exception as e:
        # EXACT ERROR PAKADNE KA LOGIC
        error_details = str(e)
        print(f"❌ GEMINI ERROR: {error_details}")
        return f"DEBUG ERROR: {error_details}"