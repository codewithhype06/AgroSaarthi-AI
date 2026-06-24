# FILE: rag_chatbot.py
# PATH: AgroSaarthi_AI/backend_api/app/rag_chatbot.py
# PURPOSE: GenAI Chatbot using ultra-fast Groq (LLaMA-3)

import os
from groq import Groq

# Fetch the secret API key from Render Environment Variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def get_agronomist_response(disease_name: str, user_query: str) -> str:
    if not GROQ_API_KEY:
        return "System Warning: GROQ_API_KEY environment variable missing."

    # Initialize Groq Client
    client = Groq(api_key=GROQ_API_KEY)
    
    # System Prompt to set the persona
    system_prompt = """
    You are 'AgroSaarthi', an expert AI Agronomist working in India.
    Your job is to provide treatment plans for crop diseases.
    
    CRITICAL RULES:
    1. You MUST reply ONLY in simple Hinglish (Hindi written in English alphabet).
    2. Start your response with "Namaste Kisan Bhai!".
    3. Keep it brief, actionable, and include chemical/organic names if necessary.
    """
    
    user_prompt = f"Disease detected: '{disease_name}'. Farmer's question: '{user_query}'"

    try:
        # Call Groq's lightning-fast LLaMA-3 model
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-8b-8192", # Extremely fast and smart model
            temperature=0.5,
            max_tokens=250,
        )
        
        # Extract and return the response
        return chat_completion.choices[0].message.content.strip()
        
    except Exception as e:
        return f"DEBUG ERROR (Groq API): {str(e)}"