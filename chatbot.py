import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ["AIzaSyCmuoXVnxbQ2Dt26bLHF5SgxAMtcL6eFhg"])

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
        role = "user" if message["role"] == "user" else "model"
        gemini_history.append(
            types.Content(role=role, parts=[types.Part(text=message["content"])])
        )

    # Add current user message
    gemini_history.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction=system_context),
        contents=gemini_history
    )

    return response.text
```
