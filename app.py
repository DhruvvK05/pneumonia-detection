import os
from chatbot import get_chatbot_response

# 🔹 Silence TensorFlow logs
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

# 🔹 Config
MODEL_PATH = "models/final_model.keras"
IMG_SIZE = 224
CLASS_NAMES = ['NORMAL', 'bacterial', 'viral']

# 🔹 Page
st.set_page_config(page_title="Pneumonia AI", layout="centered")

# 🔹 Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# 🔹 Header
st.markdown(
    "<h1 style='text-align:center;'>🩺 Pneumonia Detection AI</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:gray;'>Upload a Chest X-ray and chat with AI</p>",
    unsafe_allow_html=True
)

st.divider()

# 🔹 Upload
uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "png", "jpeg"])
predict_btn = st.button("🔍 Predict")

# 🔹 Prediction
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    st.image(img, caption="Uploaded Image", width=300)

    if predict_btn:
        with st.spinner("⚡ Analyzing..."):
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            prediction = model.predict(img_array)

            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = float(np.max(prediction) * 100)

            st.session_state.predicted_class = predicted_class
            st.session_state.confidence = confidence

        st.divider()
        st.subheader("🧾 Result")

        col1, col2 = st.columns(2)
        col1.metric("Prediction", predicted_class)
        col2.metric("Confidence", f"{confidence:.1f}%")

        st.progress(int(confidence))

else:
    st.info("👆 Upload image first")

# ================= CHAT =================
st.divider()
st.subheader("💬 AI Assistant")

st.caption("Educational use only • Not medical advice")

# Init chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Block until prediction
if "predicted_class" not in st.session_state:
    st.info("Run prediction to enable chatbot")

else:
    # 🔹 Chat display
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"""
            <div style='text-align:right; background:#DCF8C6;
            padding:10px; border-radius:10px; margin:5px'>
            {msg}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='text-align:left; background:#F1F0F0;
            padding:10px; border-radius:10px; margin:5px'>
            {msg}
            </div>
            """, unsafe_allow_html=True)

    # 🔹 Input form (FAST + clean)
    with st.form("chat_form", clear_on_submit=True):

        user_input = st.text_input(
            "Type message and press Enter",
            placeholder="Ask about your result..."
        )

        submitted = st.form_submit_button("Send")

        if submitted and user_input:

            st.session_state.chat_history.append(("user", user_input))

            with st.spinner("⚡ Thinking..."):
                reply = get_chatbot_response(
                    user_input,
                    st.session_state.predicted_class,
                    st.session_state.confidence
                )

            st.session_state.chat_history.append(("ai", reply))

            st.rerun()