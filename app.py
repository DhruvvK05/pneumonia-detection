import os
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

# 🔹 Page settings
st.set_page_config(page_title="Pneumonia AI", layout="centered")

# 🔹 Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# 🔹 Title
st.markdown(
    "<h1 style='text-align: center;'>🩺 Pneumonia Detection AI</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray;'>Upload a Chest X-ray and let AI analyze it</p>",
    unsafe_allow_html=True
)

st.divider()

# 🔹 Upload
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

# 🔹 Predict button
predict_btn = st.button("🔍 Predict")

# 🔹 Run only when button clicked
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    st.image(img, caption="Uploaded Image", width=300)

    if predict_btn:

        with st.spinner("🧠 Analyzing X-ray..."):

            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            prediction = model.predict(img_array)

            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = float(np.max(prediction) * 100)

            # 🔹 Label extraction
            file_name = uploaded_file.name.lower()

            if "bacteria" in file_name:
                actual_label = "bacterial"
            elif "virus" in file_name:
                actual_label = "viral"
            elif file_name.startswith("im"):
                actual_label = "NORMAL"
            else:
                actual_label = "Unknown"

        # 🔥 RESULTS UI
        st.divider()
        st.subheader("🧾 Diagnosis Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("🧠 Prediction", predicted_class)

        with col2:
            st.metric("📊 Confidence", f"{confidence:.2f}%")

        # 🔹 Confidence bar
        st.progress(int(confidence))

        # 🔹 Actual label
        st.write(f"🧾 Actual (from filename): **{actual_label}**")

        # 🔹 Result indicator
        if actual_label != "Unknown":
            if predicted_class == actual_label:
                st.success("✅ Correct Prediction")
            else:
                st.error("❌ Incorrect Prediction")
        else:
            st.warning("⚠️ Could not detect label from filename")

else:
    st.info("👆 Upload an image to get started")