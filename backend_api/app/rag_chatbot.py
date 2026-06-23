# FILE: rag_chatbot.py
# PATH: AgroSaarthi_AI/backend_api/app/rag_chatbot.py
# PURPOSE: GenAI and RAG Chatbot Logic for Agronomy

from langchain_core.prompts import PromptTemplate

def get_agronomist_response(disease_name: str, user_query: str) -> str:
    """
    Simulates a RAG pipeline. Formats the prompt and returns a tailored response.
    """
    # Step 1: Define how the AI should think (Prompt Engineering)
    template = """
    You are 'AgroSaarthi', an expert AI Agronomist working in India.
    A farmer is asking about the following crop disease: {disease}
    Their specific question is: {query}
    
    Please provide a step-by-step treatment plan in simple Hinglish. 
    Include organic and chemical methods.
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # Step 2: Format the prompt with actual data
    formatted_prompt = prompt.format(disease=disease_name, query=user_query)
    
    # Step 3: DUMMY LLM RESPONSE
    # In future phases, we will pass 'formatted_prompt' to Gemini/OpenAI API.
    # For now, we return a hardcoded high-quality Hinglish response for the App UI.
    
    dummy_ai_response = f"""
    Namaste Kisan Bhai! Aapke fasal mein '{disease_name}' ke lakshan dikh rahe hain. 
    Ghabrayein nahi, iska ilaaj sambhav hai. 
    
    Organic Tareeqa:
    1. Neem oil (10,000 ppm) ka spray karein (5ml per litre paani mein).
    2. Khet mein paani jama na hone dein, proper drainage banayein.
    
    Chemical Tareeqa:
    1. Mancozeb 75% WP (2 gram per litre) ka chidkav karein.
    
    Aapka sawaal tha: "{user_query}". 
    Mera sujhav hai ki turant infected patto (leaves) ko tod kar khet se door jala dein taaki bimari aage na faile.
    """
    
    return dummy_ai_response.strip()