import os
from datetime import datetime, timezone, date
import firebase_admin
from firebase_admin import credentials, auth, firestore
from fastapi import HTTPException, Header
from typing import Optional

# ─────────────────────────────────────────────
# FIREBASE ADMIN INIT (runs once)
# ─────────────────────────────────────────────
if not firebase_admin._apps:
    # On Render: set GOOGLE_APPLICATION_CREDENTIALS env var
    # pointing to your downloaded service account JSON file path
    # OR set FIREBASE_SERVICE_ACCOUNT_JSON env var with the JSON content as a string
    import json

    service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if service_account_json:
        service_account_info = json.loads(service_account_json)
        cred = credentials.Certificate(service_account_info)
    else:
        # fallback: file path
        cred = credentials.Certificate(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json"))

    firebase_admin.initialize_app(cred)

db = firestore.client()

# ─────────────────────────────────────────────
# PLAN LIMITS
# ─────────────────────────────────────────────
PLAN_LIMITS = {
    "free": {
        "weekly_analyses":  7,
        "daily_sage":       17,
        "weekly_photos":    3,
    },
    "pro": {
        "weekly_analyses":  18,
        "daily_sage":       30,
        "weekly_photos":    15,
    },
    "super_pro": {
        "weekly_analyses":  None,   # unlimited
        "daily_sage":       None,   # unlimited
        "weekly_photos":    None,   # unlimited
    },
}

# ─────────────────────────────────────────────
# HELPERS — week/day keys
# ─────────────────────────────────────────────
def get_week_start() -> str:
    """Monday of current week as YYYY-MM-DD."""
    today = date.today()
    monday = today - __import__('datetime').timedelta(days=today.weekday())
    return monday.isoformat()

def get_today() -> str:
    return date.today().isoformat()

# ─────────────────────────────────────────────
# VERIFY FIREBASE ID TOKEN → returns uid
# ─────────────────────────────────────────────
def verify_token(authorization: str) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    try:
        decoded = auth.verify_id_token(token)
        return decoded["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired Firebase token.")

# ─────────────────────────────────────────────
# GET USER DOC — auto-resets stale counters
# ─────────────────────────────────────────────
def get_user_doc(uid: str) -> dict:
    ref = db.collection("users").document(uid)
    snap = ref.get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="User document not found. Please log in again.")

    data = snap.to_dict()
    updates = {}
    current_week = get_week_start()
    current_day  = get_today()

    # Reset weekly counters if new week
    if data.get("weekStartDate") != current_week:
        updates["weekStartDate"]   = current_week
        updates["weeklyAnalyses"]  = 0
        updates["weeklyPhotos"]    = 0

    # Reset daily Sage counter if new day
    if data.get("sageDayDate") != current_day:
        updates["sageDayDate"]      = current_day
        updates["dailySageMessages"] = 0

    if updates:
        ref.update(updates)
        data.update(updates)

    return data

# ─────────────────────────────────────────────
# CHECK + INCREMENT — Resume Analysis
# ─────────────────────────────────────────────
def check_and_increment_analysis(uid: str):
    data   = get_user_doc(uid)
    plan   = data.get("plan", "free")
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    limit  = limits["weekly_analyses"]

    if limit is not None:
        used = data.get("weeklyAnalyses", 0)
        if used >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Weekly analysis limit reached ({limit}/week on {plan.replace('_',' ').title()} plan). Upgrade to get more."
            )
        db.collection("users").document(uid).update({"weeklyAnalyses": firestore.Increment(1)})

# ─────────────────────────────────────────────
# CHECK + INCREMENT — Sage Chat
# ─────────────────────────────────────────────
def check_and_increment_sage(uid: str):
    data   = get_user_doc(uid)
    plan   = data.get("plan", "free")
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    limit  = limits["daily_sage"]

    if limit is not None:
        used = data.get("dailySageMessages", 0)
        if used >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Daily Sage message limit reached ({limit}/day on {plan.replace('_',' ').title()} plan). Upgrade to get more."
            )
        db.collection("users").document(uid).update({"dailySageMessages": firestore.Increment(1)})

# ─────────────────────────────────────────────
# CHECK + INCREMENT — Photo Analysis
# ─────────────────────────────────────────────
def check_and_increment_photo(uid: str):
    data   = get_user_doc(uid)
    plan   = data.get("plan", "free")
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    limit  = limits["weekly_photos"]

    if limit is not None:
        used = data.get("weeklyPhotos", 0)
        if used >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Weekly photo analysis limit reached ({limit}/week on {plan.replace('_',' ').title()} plan). Upgrade to get more."
            )
        db.collection("users").document(uid).update({"weeklyPhotos": firestore.Increment(1)})
