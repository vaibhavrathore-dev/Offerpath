import os
import hmac
import hashlib
import razorpay
from fastapi import HTTPException

# ─────────────────────────────────────────────
# RAZORPAY CLIENT
# ─────────────────────────────────────────────
RAZORPAY_KEY_ID     = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")

client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# ─────────────────────────────────────────────
# VALID AMOUNTS (paise) — acts as server-side price guard
# No one can manipulate the amount from frontend
# ─────────────────────────────────────────────
VALID_ORDERS = {
    "pro_monthly":    7000,    # ₹70
    "pro_yearly":     72000,   # ₹720
    "super_monthly":  14000,   # ₹140
    "super_yearly":   140000,  # ₹1400
}

# Maps order type → Firestore plan value
PLAN_MAP = {
    "pro_monthly":   "pro",
    "pro_yearly":    "pro",
    "super_monthly": "super_pro",
    "super_yearly":  "super_pro",
}

# ─────────────────────────────────────────────
# CREATE ORDER — called before opening Razorpay checkout
# ─────────────────────────────────────────────
def create_razorpay_order(order_type: str, uid: str) -> dict:
    if order_type not in VALID_ORDERS:
        raise HTTPException(status_code=400, detail="Invalid plan selected.")

    amount = VALID_ORDERS[order_type]

    try:
        order = client.order.create({
            "amount":   amount,
            "currency": "INR",
            "receipt":  f"{uid[:12]}_{order_type}",
            "notes": {
                "uid":        uid,
                "order_type": order_type,
                "plan":       PLAN_MAP[order_type]
            }
        })
        return {
            "order_id":  order["id"],
            "amount":    amount,
            "currency":  "INR",
            "key_id":    RAZORPAY_KEY_ID,
            "order_type": order_type,
            "plan":      PLAN_MAP[order_type]
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to create payment order: {str(e)}")

# ─────────────────────────────────────────────
# VERIFY SIGNATURE — Razorpay HMAC SHA256 check
# ─────────────────────────────────────────────
def verify_razorpay_signature(order_id: str, payment_id: str, signature: str) -> bool:
    message = f"{order_id}|{payment_id}"
    expected = hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
