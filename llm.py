import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

def get_improvement_suggestions(resume_text, job_description, missing_skills):
    prompt = f"""
    You are a professional resume coach and HR expert.
    
    Resume: {resume_text[:2000]}
    Job Description: {job_description[:1000]}
    Missing skills: {", ".join(missing_skills[:10])}
    
    Give exactly 5 specific, actionable bullet points to improve this resume
    for this specific job. Be direct. No generic advice.
    """
    response = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

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
    response = client.chat.completions.create(
        model="google/gemma-3-27b-it:free",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
