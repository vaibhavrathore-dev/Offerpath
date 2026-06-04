import os
import re
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CLIENT SETUP
# First commit from Ubuntu 🚀
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
                continue
            raise e

    # Groq failed — fallback to Gemini
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"All AI providers failed: {str(e)}")


# ─────────────────────────────────────────────
# FUNCTIONS — exactly same as before
# ─────────────────────────────────────────────
def get_match_score(resume_text, job_description):
    prompt = f"""
    Compare this resume against this job description and give a match score.
    
    Resume: {resume_text[:2000]}
    Job Description: {job_description[:1000]}
    
    Reply with ONLY a single number between 0 and 100 representing the match percentage.
    No explanation, no text, just the number.
    """
    result = call_ai(prompt)
    try:
        score = float(re.findall(r'\d+\.?\d*', result)[0])
        return round(min(max(score, 0), 100), 1)
    except:
        return 65.0


def extract_keywords(text):
    stop_words = {
        'the','a','an','and','or','but','in','on','at','to','for','of','with',
        'by','from','is','are','was','were','be','been','have','has','had','do',
        'does','did','will','would','could','should','may','might','this','that',
        'these','those','i','we','you','he','she','it','they','our','your','his',
        'her','its','their'
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = set()
    for word in words:
        if word not in stop_words:
            keywords.add(word)
    return keywords


def get_missing_skills(resume_text, job_description):
    prompt = f"""
    Look at this resume and job description.
    List the technical skills and keywords that appear in the job description but are MISSING from the resume.
    
    Resume: {resume_text[:2000]}
    Job Description: {job_description[:1000]}
    
    Reply with ONLY a comma-separated list of missing skills/keywords.
    Maximum 15 items. No explanation, no numbering, just the comma-separated list.
    Example: docker, kubernetes, react, typescript, aws
    """
    result = call_ai(prompt)
    skills = [s.strip().lower() for s in result.split(',') if s.strip()]
    return skills[:15]


def get_matched_skills(resume_text, job_description):
    prompt = f"""
    Look at this resume and job description.
    List the technical skills and keywords that appear in BOTH the resume AND the job description.
    
    Resume: {resume_text[:2000]}
    Job Description: {job_description[:1000]}
    
    Reply with ONLY a comma-separated list of matched skills/keywords.
    Maximum 15 items. No explanation, no numbering, just the comma-separated list.
    Example: python, django, sql, git, rest api
    """
    result = call_ai(prompt)
    skills = [s.strip().lower() for s in result.split(',') if s.strip()]
    return skills[:15]
