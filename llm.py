import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")

def get_improvement_suggestions(resume_text, job_description, missing_skills):
    prompt = f"""
    You are a professional resume coach and HR expert.
    
    Resume: {resume_text[:2000]}
    Job Description: {job_description[:1000]}
    Missing skills: {", ".join(missing_skills[:10])}
    
    Give exactly 5 specific, actionable bullet points to improve this resume
    for this specific job. Be direct. No generic advice.
    """
    response = model.generate_content(prompt)
    return response.text

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
    response = model.generate_content(prompt)
    return response.text