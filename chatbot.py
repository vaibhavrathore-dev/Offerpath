import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)
MODELS = [
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3n-e4b-it:free",
    "google/gemma-3n-e2b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "nousresearch/hermes-3-405b-instruct:free",
    "qwen/qwen3-4b:free",
]

def call_ai(messages):
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                continue
            raise e
    raise Exception("All models are rate limited. Please try again later.")

def get_chatbot_response(user_message, chat_history, resume_text, job_description):
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

    return call_ai(messages)
