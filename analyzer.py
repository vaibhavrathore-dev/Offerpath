import spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_match_score(resume_text, job_description):
    resume_vector = embedding_model.encode(resume_text, convert_to_tensor=True)
    jd_vector = embedding_model.encode(job_description, convert_to_tensor=True)
    similarity = util.cos_sim(resume_vector, jd_vector)
    score = similarity.item() * 100
    return round(score, 1)

def extract_keywords(text):
    doc = nlp(text.lower())
    keywords = set()
    for token in doc:
        if not token.is_stop and token.is_alpha and len(token.text) > 2:
            keywords.add(token.lemma_)  # lemma = root form (running -> run)
    return keywords

def get_missing_skills(resume_text, job_description):
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)
    missing = jd_keywords - resume_keywords  # set subtraction
    return sorted(list(missing))[:20]

def get_matched_skills(resume_text, job_description):
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)
    matched = resume_keywords & jd_keywords  # set intersection
    return sorted(list(matched))[:20]