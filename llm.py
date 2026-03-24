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
def call_ai(prompt: str) -> str:

    # Try Groq first (fastest)
    for model in GROQ_MODELS:
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                continue  # try next Groq model
            raise e

    # Groq failed — fallback to Gemini
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"All AI providers failed: {str(e)}")


# ─────────────────────────────────────────────
# FUNCTIONS (same as before, no changes needed)
# ─────────────────────────────────────────────
def get_improvement_suggestions(resume_text, job_description, missing_skills):
    prompt = f"""
    You are a professional resume coach and HR expert.
    
    Resume: {resume_text[:2000]}
    Job Description: {job_description[:1000]}
    Missing skills: {", ".join(missing_skills[:10])}
    
    Give exactly 5 specific, actionable bullet points to improve this resume
    for this specific job. Be direct. No generic advice.
    """
    return call_ai(prompt)


def get_resume_score_breakdown(resume_text):
    prompt = f"""
    You are a senior HR recruiter with 10 years of experience.
    Score these resume sections out of 10 and explain each in one sentence:
    - Summary/Objective
    - Work Experience
    - Skills
    - Education
    - Overall formatting
    
    Resume: {resume_text[:2000]}
    """
    return call_ai(prompt)
