import os
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from ultralytics import YOLO

# -------------------------------------------------
# Streamlit Page
# -------------------------------------------------
st.set_page_config(
    page_title="🚨 SafeFall AI",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 SafeFall AI - Elderly Fall Detection System")
st.caption("YOLOv8 Pose + Deep Learning Activity Classification")

# -------------------------------------------------
# Load YOLO Pose Model
# -------------------------------------------------
MODEL_PATH = "model/yolov8n-pose.pt"

@st.cache_resource
def load_pose_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    else:
        st.warning("YOLO pose model not found. Downloading once...")
        return YOLO("yolov8n-pose.pt")

pose_model = load_pose_model()

# -------------------------------------------------
# Load Trained AI Classifier
# -------------------------------------------------
@st.cache_resource
def load_classifier():
    classifier = tf.keras.models.load_model("model/activity_classifier.keras")
    scaler = joblib.load("model/scaler.pkl")
    encoder = joblib.load("model/label_encoder.pkl")
    return classifier, scaler, encoder

try:
    classifier, scaler, encoder = load_classifier()
except Exception:
    st.error("❌ Trained model files are missing.")
    st.info("""
Create a **model** folder containing:

- activity_classifier.keras
- scaler.pkl
- label_encoder.pkl
- yolov8n-pose.pt
""")
    st.stop()

# -------------------------------------------------
# Prediction Function
# -------------------------------------------------
def predict_frame(frame):
    results = pose_model(frame, verbose=False)

    annotated = results[0].plot()

    if results[0].keypoints is None or results[0].keypoints.xy is None:
        return annotated, "No Person", 0

    keypoints = results[0].keypoints.xy

    if len(keypoints) == 0:
        return annotated, "No Person", 0

    features = keypoints[0].cpu().numpy().flatten()

    features = scaler.transform([features])

    prediction = classifier.predict(features, verbose=False)[0]

    label = encoder.inverse_transform([np.argmax(prediction)])[0]
    confidence = float(np.max(prediction))

    return annotated, label, confidence

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("📂 Detection Mode")

mode = st.sidebar.radio(
    "Choose Input",
    ["🖼️ Image Detection", "🎥 Video Detection"]
)

# -------------------------------------------------
# IMAGE DETECTION
# -------------------------------------------------
if mode == "🖼️ Image Detection":

    uploaded_image = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
        frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        output, label, confidence = predict_frame(frame)

        st.image(
            cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        col1.metric("Activity", label.title())
        col2.metric("Confidence", f"{confidence:.2%}")

        if label.lower() == "fall" and confidence > 0.60:
            st.error("🚨 FALL DETECTED — Notify Caregiver Immediately!")
        else:
            st.success("✅ Normal Activity Detected")

# -------------------------------------------------
# VIDEO DETECTION
# -------------------------------------------------
if mode == "🎥 Video Detection":

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture("temp_video.mp4")

        frame_placeholder = st.empty()

        activities = []

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            output, label, confidence = predict_frame(frame)

            activities.append(label)

            cv2.putText(
                output,
                f"{label.upper()} ({confidence:.2%})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            frame_placeholder.image(
                cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        cap.release()

        st.success("🎉 Video Analysis Completed")

        st.subheader("📊 Activity Analytics")

        counts = pd.Series(activities).value_counts()

        st.bar_chart(counts)

        col1, col2 = st.columns(2)

        col1.metric("Total Frames Analysed", len(activities))
        col2.metric("Fall Frames", activities.count("fall"))

        if activities.count("fall") > 0:
            st.error("🚨 Emergency Alert: Fall detected in uploaded video.")
        else:
            st.success("✅ No fall detected.")
