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
        return "⚠️ AI is busy, try again."