import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import io

# Import our existing backend modules
from parser import extract_text_from_pdf
from analyzer import get_match_score, get_missing_skills, get_matched_skills
from llm import get_improvement_suggestions, get_resume_score_breakdown
from chatbot import get_chatbot_response

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Offerpath API",
    description="Backend API for Offerpath — AI Resume Analyzer",
    version="1.0.0"
)

# Allow frontend HTML files to call this API
# During development: allow all origins
# In production: replace * with your actual frontend domain
app.add_middleware(
    CORSMiddleware,
   allow_origins=["https://offerpath.co.in"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# MODELS (request/response shapes)
# ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: List[ChatMessage] = []
    resume_text: str
    job_description: Optional[str] = ""

class AnalyzeResponse(BaseModel):
    match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    suggestions: str
    breakdown: str
    resume_text: str

class ChatResponse(BaseModel):
    reply: str

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

# Health check — to verify API is running
@app.get("/")
def root():
    return {
        "status": "Offerpath API is running",
        "version": "1.0.0",
        "endpoints": ["/analyze", "/chat", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# /analyze — Core resume analysis endpoint
# Called by analyzer.html when user clicks "Analyze my resume"
# ─────────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    resume: UploadFile = File(...),          # PDF file from frontend
    job_description: str = Form(...)         # Job description text from frontend
):
    # Validate file type
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Validate job description length
    if len(job_description.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Job description is too short. Please paste the full job description."
        )

    try:
        # Step 1 — Extract text from PDF
        # We need to wrap the file bytes in an object that has a .read() method
        # because our parser.py was written for Streamlit's file uploader
        file_bytes = await resume.read()

        class FileWrapper:
            def read(self):
                return file_bytes

        resume_text = extract_text_from_pdf(FileWrapper())

        if not resume_text or len(resume_text.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the PDF. Make sure it's not a scanned image."
            )

        # Step 2 — Calculate match score
        match_score = get_match_score(resume_text, job_description)

        # Step 3 — Extract matched and missing skills
        matched_skills = get_matched_skills(resume_text, job_description)
        missing_skills = get_missing_skills(resume_text, job_description)

        # Step 4 — Generate AI suggestions via Gemini
        suggestions = get_improvement_suggestions(
            resume_text,
            job_description,
            missing_skills
        )

        # Step 5 — Generate section breakdown via Gemini
        breakdown = get_resume_score_breakdown(resume_text)

        return AnalyzeResponse(
            match_score=match_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            suggestions=suggestions,
            breakdown=breakdown,
            resume_text=resume_text   # returned so frontend can pass it to chatbot
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


# ─────────────────────────────────────────────
# /chat — Sage chatbot endpoint
# Called by chatbot.html every time user sends a message
# ─────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    if not request.resume_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No resume text found. Please analyze your resume first."
        )

    try:
        # Convert chat history to the format chatbot.py expects
        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.chat_history
        ]

        # Get Sage's response from Gemini
        reply = get_chatbot_response(
            user_message=request.message,
            chat_history=chat_history,
            resume_text=request.resume_text,
            job_description=request.job_description
        )

        return ChatResponse(reply=reply)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sage failed to respond: {str(e)}"
        )


# ─────────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False
    )
