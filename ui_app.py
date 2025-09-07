import streamlit as st
from PIL import Image
import pandas as pd
from src.inference import efficientnet_inference


st.set_page_config(page_title="Food Classification", page_icon="🍽️", layout="centered")

st.markdown("<h1 style='text-align: center; color: #FF7043;'>🍽️ Food Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a food image and let the model predict its class with confidence.</p>", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Define class names
class_names = ["pizza", "steak", "sushi"]

if uploaded_file is not None:
    # Run inference
    with st.spinner("🔍 Classifying..."):
        pred_class, confidence, probabilities = efficientnet_inference(uploaded_file, class_names=class_names)

    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    # Show results
    st.success(f"✅ Prediction: **{pred_class}** with **{confidence*100:.2f}%** confidence")

    # Show class probabilities as bar chart
    # Show all probabilities
    st.subheader("📊 Class Probabilities")
    for i, (cls, prob) in enumerate(zip(class_names, probabilities)):
        st.write(f"{cls}: {prob:.2%}")

else:
    st.info("⬅️ Upload an image from the sidebar to get started.")
