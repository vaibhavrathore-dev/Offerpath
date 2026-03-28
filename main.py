import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from parser import extract_text_from_pdf
from analyzer import get_match_score, get_missing_skills, get_matched_skills
from llm import get_improvement_suggestions, get_resume_score_breakdown
from chatbot import get_chatbot_response
from photo_analyzer import router as photo_router
from firebase_utils import verify_token, check_and_increment_analysis, check_and_increment_sage, check_and_increment_photo, db
from razorpay_utils import create_razorpay_order, verify_razorpay_signature, PLAN_MAP

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(title="Offerpath API", version="2.0.0")
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

class CreateOrderRequest(BaseModel):
    order_type: str  # pro_monthly | pro_yearly | super_monthly | super_yearly

class VerifyPaymentRequest(BaseModel):
    razorpay_order_id:   str
    razorpay_payment_id: str
    razorpay_signature:  str
    order_type:          str

# ─────────────────────────────────────────────
# BASIC ROUTES
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Offerpath API is running", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ─────────────────────────────────────────────
# /create-order — Step 1: create Razorpay order
# Amount is set SERVER-SIDE — frontend can't manipulate it
# ─────────────────────────────────────────────
@app.post("/create-order")
async def create_order(
    request: CreateOrderRequest,
    authorization: Optional[str] = Header(None)
):
    uid = verify_token(authorization)
    order = create_razorpay_order(request.order_type, uid)
    return order

# ─────────────────────────────────────────────
# /verify-payment — Step 2: verify + activate plan
# ─────────────────────────────────────────────
@app.post("/verify-payment")
async def verify_payment(
    request: VerifyPaymentRequest,
    authorization: Optional[str] = Header(None)
):
    # 1. Auth check
    uid = verify_token(authorization)

    # 2. Signature verification — prevents fake payment claims
    is_valid = verify_razorpay_signature(
        request.razorpay_order_id,
        request.razorpay_payment_id,
        request.razorpay_signature
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail="Payment verification failed. Invalid signature.")

    # 3. Get plan name from order type
    plan = PLAN_MAP.get(request.order_type)
    if not plan:
        raise HTTPException(status_code=400, detail="Invalid order type.")

    # 4. Activate plan in Firestore
    try:
        from datetime import datetime, timezone
        db.collection("users").document(uid).update({
            "plan":            plan,
            "paymentId":       request.razorpay_payment_id,
            "orderId":         request.razorpay_order_id,
            "orderType":       request.order_type,
            "planActivatedAt": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate plan: {str(e)}")

    return {
        "success": True,
        "plan":    plan,
        "message": f"{plan.replace('_',' ').title()} plan activated!"
    }

# ─────────────────────────────────────────────
# /analyze
# ─────────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    authorization: Optional[str] = Header(None)
):
    uid = verify_token(authorization)
    check_and_increment_analysis(uid)

    if not resume.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if len(job_description.strip()) < 50:
        raise HTTPException(status_code=400, detail="Job description is too short.")

    try:
        file_bytes = await resume.read()

        class FileWrapper:
            def read(self): return file_bytes

        resume_text = extract_text_from_pdf(FileWrapper())
        if not resume_text or len(resume_text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        match_score    = get_match_score(resume_text, job_description)
        matched_skills = get_matched_skills(resume_text, job_description)
        missing_skills = get_missing_skills(resume_text, job_description)
        suggestions    = get_improvement_suggestions(resume_text, job_description, missing_skills)
        breakdown      = get_resume_score_breakdown(resume_text)

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
async def chat(
    request: ChatRequest,
    authorization: Optional[str] = Header(None)
):
    uid = verify_token(authorization)
    check_and_increment_sage(uid)

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="No resume text found.")

    try:
        chat_history = [{"role": m.role, "content": m.content} for m in request.chat_history]
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
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
