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
from photo_analyzer import router as photo_router  # ← photo analyzer

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Offerpath API",
    description="Backend API for Offerpath — AI Resume Analyzer",
    version="1.0.0"
)

# ← router registered AFTER app is created
app.include_router(photo_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://offerpath.co.in",
        "https://www.offerpath.co.in",
        "https://offerpath-4e52f.web.app",
        "https://offerpath-4e52f.firebaseapp.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
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

@app.get("/")
def root():
    return {
        "status": "Offerpath API is running",
        "version": "1.0.0",
        "endpoints": ["/analyze", "/chat", "/health", "/api/analyze-photo"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# ─────────────────────────────────────────────
# /analyze
# ─────────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if len(job_description.strip()) < 50:
        raise HTTPException(status_code=400, detail="Job description is too short. Please paste the full job description.")

    try:
        file_bytes = await resume.read()

        class FileWrapper:
            def read(self):
                return file_bytes

        resume_text = extract_text_from_pdf(FileWrapper())

        if not resume_text or len(resume_text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF. Make sure it's not a scanned image.")

        match_score     = get_match_score(resume_text, job_description)
        matched_skills  = get_matched_skills(resume_text, job_description)
        missing_skills  = get_missing_skills(resume_text, job_description)
        suggestions     = get_improvement_suggestions(resume_text, job_description, missing_skills)
        breakdown       = get_resume_score_breakdown(resume_text)

        return AnalyzeResponse(
            match_score=match_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            suggestions=suggestions,
            breakdown=breakdown,
            resume_text=resume_text
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ─────────────────────────────────────────────
# /chat
# ─────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="No resume text found. Please analyze your resume first.")

    try:
        chat_history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        reply = get_chatbot_response(
            user_message=request.message,
            chat_history=chat_history,
            resume_text=request.resume_text,
            job_description=request.job_description
        )
        return ChatResponse(reply=reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sage failed to respond: {str(e)}")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False
    )
