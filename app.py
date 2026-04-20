import os
from chatbot import get_chatbot_response

# 🔹 Silence TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ================= STYLES =================
st.markdown("""
<style>
* { font-family: 'Inter', sans-serif !important; }

.user-bubble {
    background:#DCF8C6;
    color:#000;
    padding:14px 18px;
    border-radius:14px;
    max-width:70%;
    font-size:16px;
    line-height:1.6;
    font-weight:500;
}

.ai-bubble {
    background:#1f1f1f;
    color:#fff;
    padding:14px 18px;
    border-radius:14px;
    max-width:70%;
    font-size:16px;
    line-height:1.6;
}

div[data-baseweb="input"] input {
    font-size:16px !important;
}

button[kind="secondary"] {
    font-size:15px !important;
    border-radius:10px !important;
}

h1 { font-size:36px !important; }
h3 { font-size:22px !important; }
</style>
""", unsafe_allow_html=True)

# ================= CONFIG =================
MODEL_PATH = "models/final_model.keras"
IMG_SIZE = 224
CLASS_NAMES = ['NORMAL', 'bacterial', 'viral']

st.set_page_config(page_title="Pneumonia AI", layout="centered")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ================= HEADER =================
st.markdown("<h1 style='text-align:center;'>🩺 Pneumonia Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Upload X-ray & chat with AI</p>", unsafe_allow_html=True)

st.divider()

# ================= UPLOAD =================
uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg","png","jpeg"])
predict_btn = st.button("🔍 Predict")

# ================= PREDICTION =================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    st.image(img, width=300)

    if predict_btn:
        with st.spinner("⚡ Analyzing..."):
            arr = np.array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)

            pred = model.predict(arr)
            predicted_class = CLASS_NAMES[np.argmax(pred)]
            confidence = float(np.max(pred) * 100)

            st.session_state.predicted_class = predicted_class
            st.session_state.confidence = confidence

        st.subheader("🧾 Result")
        col1, col2 = st.columns(2)
        col1.metric("Prediction", predicted_class)
        col2.metric("Confidence", f"{confidence:.1f}%")
        st.progress(int(confidence))

else:
    st.info("Upload image to start")

# ================= CHAT =================
st.divider()
st.subheader("💬 AI Assistant")
st.caption("Educational use only • Not medical advice")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "predicted_class" not in st.session_state:
    st.info("Run prediction to enable chatbot")

else:
    # 🔹 Display chat
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"""
            <div style='display:flex; justify-content:flex-end; margin-bottom:10px;'>
                <div class='user-bubble'>{msg}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='display:flex; justify-content:flex-start; margin-bottom:10px;'>
                <div class='ai-bubble'>{msg}</div>
            </div>
            """, unsafe_allow_html=True)

    # 🔹 Quick buttons
    col1, col2, col3 = st.columns(3)

    if col1.button("Explain Result"):
        st.session_state.quick = "Explain my result"

    if col2.button("Symptoms"):
        st.session_state.quick = "What are symptoms?"

    if col3.button("Precautions"):
        st.session_state.quick = "What precautions should I take?"

    # 🔹 Input form
    with st.form("chat_form", clear_on_submit=True):

        default = st.session_state.get("quick", "")
        user_input = st.text_input("Type message...", value=default)

        send = st.form_submit_button("Send")

        if send and user_input:

            st.session_state.chat_history.append(("user", user_input))
            st.session_state.quick = ""

            with st.spinner("⚡ Thinking..."):
                reply = get_chatbot_response(
                    user_input,
                    st.session_state.predicted_class,
                    st.session_state.confidence
                )

            st.session_state.chat_history.append(("ai", reply))
            st.rerun()