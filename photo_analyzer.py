import os
import cv2
import numpy as np
import base64
import json
from io import BytesIO
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException
from groq import Groq
import google.generativeai as genai

router = APIRouter()

# ── Reuse same clients (or import your call_ai from ai_core.py)
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_vision_model = genai.GenerativeModel("gemini-1.5-flash")

# ─────────────────────────────────────────────
# SIGNAL 1 — Sharpness (Laplacian variance)
# ─────────────────────────────────────────────
def compute_sharpness(img_cv) -> int:
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # < 100 = blurry, > 500 = sharp
    return max(0, min(100, int(variance / 5)))

# ─────────────────────────────────────────────
# SIGNAL 2 — Lighting (HSV brightness analysis)
# ─────────────────────────────────────────────
def compute_lighting(img_cv) -> int:
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()
    if 110 <= brightness <= 190:
        return 100
    elif brightness < 110:
        return max(0, int(brightness * 0.9))
    else:
        return max(0, int(100 - (brightness - 190) * 0.8))

# ─────────────────────────────────────────────
# SIGNAL 3 — Gemini Vision (semantic signals)
# Groq doesn't support vision, so Gemini is
# primary here with a structured JSON prompt
# ─────────────────────────────────────────────
def compute_gemini_signals(pil_image: Image.Image) -> dict:
    prompt = """You are a professional LinkedIn profile photo evaluator for Indian college students applying to tech and non-tech jobs.

Analyze this photo carefully and return ONLY a valid JSON object. No extra text, no markdown, no backticks.

{
  "face_position": <0-100>,
  "attire": <0-100>,
  "background": <0-100>,
  "grade": "<one of: Excellent · Professional | Good · Mostly Professional | Fair · Needs Improvement | Poor · Not Suitable>",
  "overall_impression": "<1-2 honest sentences>",
  "tips": ["<tip 1>", "<tip 2>", "<tip 3>", "<tip 4>"]
}

Scoring guide:
- face_position: Is exactly one face visible, centered, appropriate size (25-40% of frame), looking at camera?
- attire: Does clothing look professional for a corporate/tech job? Formal shirt, blazer = high score. T-shirt with graphics = low.
- background: Is the background clean, neutral, blurred, or office-like? Bedroom clutter, busy patterns = low score.

Be specific in tips. Address the actual problems you see in this specific photo."""

    try:
        response = gemini_vision_model.generate_content([prompt, pil_image])
        raw = response.text.strip()
        # Strip markdown fences if Gemini wraps it anyway
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        # Gemini returned non-JSON — ask Groq to extract it as text fallback
        raise Exception("Gemini returned invalid JSON")
    except Exception as e:
        raise Exception(f"Gemini vision failed: {str(e)}")

# ─────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────
@router.post("/analyze-photo")
async def analyze_photo(file: UploadFile = File(...)):
    # ── Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, or WebP images are allowed.")

    contents = await file.read()

    # ── Validate file size (max 5MB)
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be under 5MB.")

    # ── Load and resize image
    try:
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
        pil_image.thumbnail((1024, 1024))  # cap size for faster processing
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Please upload a valid photo.")

    # ── Convert to OpenCV for math-based signals
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # ── Run OpenCV signals (fast, local)
    sharpness = compute_sharpness(img_cv)
    lighting  = compute_lighting(img_cv)

    # ── Run Gemini Vision (semantic signals)
    try:
        gemini_data = compute_gemini_signals(pil_image)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    face_position = gemini_data.get("face_position", 0)
    attire        = gemini_data.get("attire", 0)
    background    = gemini_data.get("background", 0)

    # ── Weighted final score
    # Face position is the most important for a profile photo
    final_score = int(
        sharpness     * 0.15 +
        lighting      * 0.15 +
        face_position * 0.30 +
        attire        * 0.25 +
        background    * 0.15
    )

    return {
        "final_score": final_score,
        "breakdown": {
            "sharpness":      sharpness,
            "lighting":       lighting,
            "face_position":  face_position,
            "attire":         attire,
            "background":     background,
        },
        "grade":               gemini_data.get("grade", "—"),
        "overall_impression":  gemini_data.get("overall_impression", ""),
        "tips":                gemini_data.get("tips", []),
    }
