import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

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

    messages = [{"role": "system", "content": system_context}]

    for message in chat_history:
        messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=messages
    )

    return response.choices[0].message.content
