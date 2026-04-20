# chatbot.py

import os
import google.generativeai as genai

# 🔹 Setup Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM_PROMPT = """
You are a friendly AI assistant.

Explain pneumonia (normal, bacterial, viral) simply.
Give general advice and precautions.

Do NOT give medicines or prescriptions.

Keep answer very short (2-3 lines max).
End with: This is educational, not medical advice.
"""


def get_chatbot_response(user_input, predicted_class=None, confidence=None):

    context = ""
    if predicted_class:
        context = f"Prediction: {predicted_class}, Confidence: {confidence:.1f}%\n"

    prompt = SYSTEM_PROMPT + context + "\nUser: " + user_input

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return "⚠️ AI is busy, try again."# chatbot.py

import os
import time
import re
from functools import lru_cache

import google.generativeai as genai

# ---------- CONFIG ----------
MODEL_NAME = "gemini-1.5-flash"   # fastest
TIMEOUT_SECONDS = 8               # keep it snappy
MAX_RETRIES = 1                   # 1 quick retry only

SYSTEM_PROMPT = """
You are a friendly AI assistant for an educational demo.

Explain pneumonia (normal, bacterial, viral) simply.
Give general precautions and when to seek care.

Do NOT give medicines or prescriptions.

Keep answers short (2–3 lines).
End with: This is educational, not medical advice.
"""

# ---------- INIT GEMINI ----------
_API_KEY = os.getenv("GEMINI_API_KEY")
_model = None
if _API_KEY:
    try:
        genai.configure(api_key=_API_KEY)
        _model = genai.GenerativeModel(MODEL_NAME)
    except Exception:
        _model = None


# ---------- LOCAL FALLBACK (ALWAYS AVAILABLE) ----------
def _local_fallback(user_input: str, predicted_class=None, confidence=None) -> str:
    ui = (user_input or "").lower()

    # tiny intent routing
    if any(k in ui for k in ["what", "mean", "explain", "why"]):
        if predicted_class:
            return (
                f"{predicted_class} means an infection pattern seen on the X-ray. "
                f"Bacterial is often localized; viral is more diffuse. "
                f"Confidence ~{confidence:.1f}%.\n"
                "This is educational, not medical advice."
            )
        return "Pneumonia is a lung infection; patterns can suggest bacterial or viral causes.\nThis is educational, not medical advice."

    if any(k in ui for k in ["symptom", "sign"]):
        return (
            "Common symptoms: cough, fever, chest pain, shortness of breath. "
            "Seek care if breathing is difficult or symptoms worsen.\n"
            "This is educational, not medical advice."
        )

    if any(k in ui for k in ["treat", "treatment", "medicine", "medication"]):
        return (
            "General care includes rest, fluids, and monitoring symptoms. "
            "Consult a doctor for proper diagnosis and treatment.\n"
            "This is educational, not medical advice."
        )

    if any(k in ui for k in ["when", "doctor", "urgent", "emergency"]):
        return (
            "Seek medical help if there is severe breathlessness, high fever, or worsening condition.\n"
            "This is educational, not medical advice."
        )

    # default
    return (
        "It looks like a lung condition pattern on the X-ray. "
        "Consider symptoms and consult a doctor if concerned.\n"
        "This is educational, not medical advice."
    )


# ---------- GEMINI CALL (SHORT + SAFE) ----------
def _call_gemini(prompt: str):
    if _model is None:
        return None, "No API key or model init failed"

    try:
        # simple timeout guard via time check
        start = time.time()
        resp = _model.generate_content(prompt)
        if (time.time() - start) > TIMEOUT_SECONDS:
            return None, "timeout"
        text = getattr(resp, "text", None)
        return text, None
    except Exception as e:
        return None, str(e)


# ---------- CACHE (instant repeat answers) ----------
@lru_cache(maxsize=128)
def _cached_response(prompt: str):
    return _call_gemini(prompt)


# ---------- MAIN API ----------
def get_chatbot_response(user_input, predicted_class=None, confidence=None):
    context = ""
    if predicted_class:
        context = f"Prediction: {predicted_class} ({confidence:.1f}%).\n"

    prompt = SYSTEM_PROMPT + context + "\nUser: " + (user_input or "")

    # 1) Try Gemini (cached)
    text, err = _cached_response(prompt)
    if text:
        return text

    # 2) Quick retry (no long backoff)
    for _ in range(MAX_RETRIES):
        text, err = _call_gemini(prompt)
        if text:
            return text

    # 3) Guaranteed fallback (instant, never empty)
    return _local_fallback(user_input, predicted_class, confidence)