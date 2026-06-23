# FILE: rag_chatbot.py
# PATH: AgroSaarthi_AI/backend_api/app/rag_chatbot.py
# PURPOSE: GenAI Chatbot Logic using Real Google Gemini API

import os
import google.generativeai as genai

# Fetch the secret API key from Render Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize Gemini if key is present
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Using the fast and capable Gemini 1.5 Flash model
    model = genai.GenerativeModel('gemini-1.5-flash')
    USE_MOCK_AI = False
else:
    USE_MOCK_AI = True
    print("⚠️ WARNING: GEMINI_API_KEY not found. Using MOCK AI response.")

def get_agronomist_response(disease_name: str, user_query: str) -> str:
    """
    Calls Google Gemini API to generate a contextual, Hinglish treatment plan.
    """
    if USE_MOCK_AI:
        return f"Namaste! Yeh ek dummy response hai kyunki Gemini API key nahi mili. Aapki bimari '{disease_name}' ke liye check karein."

    # Step 1: Prompt Engineering (Setting the AI's persona)
    prompt = f"""
    You are 'AgroSaarthi', an expert AI Agronomist working in India.
    A farmer is asking about the following crop disease: '{disease_name}'
    Their specific question is: '{user_query}'
    
    Instructions:
    1. Provide a step-by-step treatment plan.
    2. Include both organic and chemical methods if applicable.
    3. The language MUST be simple Hinglish (Hindi written in English alphabet), easily understandable by an Indian farmer.
    4. Keep it friendly and professional. Start with "Namaste Kisan Bhai/Behen!".
    """
    
    try:
        # Step 2: Call the AI
        response = model.generate_content(prompt)
        
        # Step 3: Return the generated text
        if response.text:
            return response.text.strip()
        else:
            return "Maaf kijiye, main abhi iska jawab nahi de paa raha hoon. Kripya baad mein try karein."
            
    except Exception as e:
        print(f"❌ GEMINI ERROR: {e}")
        return "Server error: AI se connect karne mein samasya aayi. Kripya apne internet ya server logs check karein."