import os
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

def get_match_score(resume_text, job_description):
    prompt = f"""
    Compare this resume against this job description and give a match score.
    
    Resume: {resume_text[:2000]}
    Job Description: {job_description[:1000]}
    
    Reply with ONLY a single number between 0 and 100 representing the match percentage.
    No explanation, no text, just the number.
    """
    response = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        score = float(re.findall(r'\d+\.?\d*', response.choices[0].message.content)[0])
        return round(min(max(score, 0), 100), 1)
    except:
        return 65.0

def extract_keywords(text):
    stop_words = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','is','are','was','were','be','been','have','has','had','do','does','did','will','would','could','should','may','might','this','that','these','those','i','we','you','he','she','it','they','our','your','his','her','its','their'}
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
    response = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[{"role": "user", "content": prompt}]
    )
    skills = [s.strip().lower() for s in response.choices[0].message.content.split(',') if s.strip()]
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
    response = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[{"role": "user", "content": prompt}]
    )
    skills = [s.strip().lower() for s in response.choices[0].message.content.split(',') if s.strip()]
    return skills[:15]
