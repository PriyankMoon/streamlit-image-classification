
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from transformers import pipeline

# Load the MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Load the zero-shot classification pipeline from HuggingFace
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# Function to decode predictions from MobileNet
def decode_prediction(preds):
    return tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]

st.title("🖼️ Free Image Classifier with LLM Integration")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and classify
    st.write("🔍 Classifying using MobileNetV2...")
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)
    label_id, label_name, confidence = decode_prediction(prediction)
    confidence = confidence*100

    st.success(f"🎯 Predicted Label: **{label_name}** (Confidence: {confidence:.2f}%)")

    # LLM explanation
    st.write("🤖 Asking LLM for description...")
    candidate_labels = ["animal", "vehicle", "food", "person", "nature", "technology", "art", "furniture", "tool"]

    llm_output = classifier(label_name, candidate_labels)

    st.write("🔎 LLM-enhanced Description:")
    st.json(llm_output)
