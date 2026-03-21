import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def get_chatbot_response(user_message, chat_history, resume_text, job_description):
    """
    chat_history is a list of dicts:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    system_context = f"""
    You are OfferPath AI, a helpful resume coach and career advisor.
    The user's resume: {resume_text[:1500]}
    The job they are applying for: {job_description[:800]}
    Answer based on their actual resume. Keep responses to 3-5 sentences max.
    """

    gemini_history = []
    for message in chat_history:
        gemini_history.append({
            "role": message["role"],
            "parts": [message["content"]]
        })

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction=system_context
    )
    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(user_message)
    return response.text