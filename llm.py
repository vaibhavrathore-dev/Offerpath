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
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free"
]

def call_ai(prompt):
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                continue
            raise e
    raise Exception("All models are rate limited. Please try again later.")

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
