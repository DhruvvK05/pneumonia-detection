# chatbot.py

import os
from dotenv import load_dotenv
import requests

# 🔥 LOAD ENV FIRST (VERY IMPORTANT)
load_dotenv()

# 🔹 Get API key
API_KEY = os.getenv("OPENROUTER_API_KEY")

# 🔹 DEBUG (remove later)
print("LOADED KEY:", API_KEY)


def get_chatbot_response(user_input, predicted_class=None, confidence=None):

    if not API_KEY:
        return "❌ API KEY NOT LOADED (dotenv issue)"

    prompt = f"""
Explain pneumonia simply.

User: {user_input}
Prediction: {predicted_class}
Confidence: {confidence}

Keep it short.
"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=10
        )

        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ Error: {str(e)}"