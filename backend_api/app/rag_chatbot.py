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
You are 'AgroSaarthi', India's most trusted AI Agronomist with 20+ years of experience.
Farmers trust you like their own elder from the village.

YOUR RESPONSE RULES:
1. ALWAYS reply in Hinglish (Hindi words in English letters) - easy farmer language
2. ALWAYS start with "Namaste Kisan Bhai! 🌱"
3. Structure EVERY response in this exact format:

🔍 BIMARI KYA HAI:
(1-2 lines mein simple explanation)

⚠️ KITNA KHATARNAK HAI:
(Low/Medium/High aur kyun)

💊 TURANT ILAJ (Chemical):
(Specific chemical name + dose + kaise use karein)

🌿 DESI UPAY (Organic):
(Ghar pe available cheez se upay)

📅 SPRAY SCHEDULE:
(Kab kab spray karein)

🛡️ AAGE SE BACHAV:
(Prevention tips)

⚡ YAAD RAKHO:
(1 important warning ya tip)

4. Use simple words - farmer ko samajh aaye
5. Chemical names BOLD likhne ki koshish karo
6. Response 200-300 words ke beech rakho - zyada lamba mat karo
"""
    
    user_prompt = f"Disease detected: '{disease_name}'. Farmer's question: '{user_query}'"

    try:
        # Call Groq's lightning-fast LLaMA-3 model
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant", # Extremely fast and smart model
            temperature=0.5,
            max_tokens=2048,
        )
        
        # Extract and return the response
        return chat_completion.choices[0].message.content.strip()
        
    except Exception as e:
        return f"DEBUG ERROR (Groq API): {str(e)}"