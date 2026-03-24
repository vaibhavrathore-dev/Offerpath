import os
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CLIENT SETUP
# ─────────────────────────────────────────────
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ─────────────────────────────────────────────
# GROQ MODELS (fastest first)
# ─────────────────────────────────────────────
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

# ─────────────────────────────────────────────
# CORE AI CALLER — Groq first, Gemini fallback
# ─────────────────────────────────────────────
def call_ai(messages: list) -> str:

    # Try Groq first (fastest)
    for model in GROQ_MODELS:
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                continue  # try next Groq model
            raise e

    # Groq failed — fallback to Gemini
    # Gemini doesn't support system role so we merge it into first user message
    try:
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        conversation = []
        for m in messages:
            if m["role"] == "system":
                continue
            elif m["role"] == "user":
                content = f"{system_msg}\n\n{m['content']}" if not conversation else m["content"]
                conversation.append({"role": "user", "parts": [content]})
            elif m["role"] == "assistant":
                conversation.append({"role": "model", "parts": [m["content"]]})

        chat = gemini_model.start_chat(history=conversation[:-1])
        response = chat.send_message(conversation[-1]["parts"][0])
        return response.text
    except Exception as e:
        raise Exception(f"All AI providers failed: {str(e)}")


# ─────────────────────────────────────────────
# SAGE CHATBOT — same as before, no changes
# ─────────────────────────────────────────────
def get_chatbot_response(user_message, chat_history, resume_text, job_description):
    system_context = f"""
    You are Sage, OfferPath's personal AI career coach.
    The user's resume: {resume_text[:1500]}
    The job they are applying for: {job_description[:800]}
    Answer based on their actual resume. Keep responses to 3-5 sentences max.
    """
    messages = [{"role": "system", "content": system_context}]

    for message in chat_history:
        messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    messages.append({"role": "user", "content": user_message})

    return call_ai(messages)
