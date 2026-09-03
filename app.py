import os
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="🚨 SafeFall AI",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 SafeFall AI - Elderly Fall Detection")
st.markdown("YOLOv8 Pose Estimation + Activity Classification")

# -------------------- Load Models --------------------
MODEL_PATH = "model/yolov8n-pose.pt"

@st.cache_resource
def load_models():
    pose = YOLO(MODEL_PATH if os.path.exists(MODEL_PATH) else "yolov8n-pose.pt")

    classifier = joblib.load("model/activity_classifier.pkl")
    scaler = joblib.load("model/scaler.pkl")
    encoder = joblib.load("model/label_encoder.pkl")

    return pose, classifier, scaler, encoder

try:
    pose_model, classifier, scaler, encoder = load_models()
except Exception as e:
    st.error("❌ Model files are missing.")
    st.code(str(e))
    st.info("""
Create this folder inside your GitHub repo:

model/
├── yolov8n-pose.pt
├── activity_classifier.pkl
├── scaler.pkl
└── label_encoder.pkl
""")
    st.stop()

# -------------------- Prediction Function --------------------
def predict_frame(frame):
    results = pose_model(frame, verbose=False)

    annotated = results[0].plot()

    if results[0].keypoints is None or results[0].keypoints.xy is None:
        return annotated, "No Person", 0.0

    if len(results[0].keypoints.xy) == 0:
        return annotated, "No Person", 0.0

    keypoints = results[0].keypoints.xy[0].cpu().numpy().flatten()

    features = scaler.transform([keypoints])

    probs = classifier.predict_proba(features)[0]
    idx = np.argmax(probs)

    label = encoder.inverse_transform([idx])[0]
    confidence = float(probs[idx])

    return annotated, label, confidence

# -------------------- Sidebar --------------------
mode = st.sidebar.radio(
    "Choose Detection Mode",
    ["🖼️ Image Detection", "🎥 Video Detection"]
)

# ==========================================================
# IMAGE DETECTION
# ==========================================================
if mode == "🖼️ Image Detection":

    image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if image:

        img = cv2.imdecode(
            np.frombuffer(image.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        output, label, confidence = predict_frame(img)

        st.image(
            cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        c1, c2 = st.columns(2)

        c1.metric("Detected Activity", label.title())
        c2.metric("Confidence", f"{confidence:.2%}")

        if label.lower() == "fall" and confidence > 0.60:
            st.error("🚨 EMERGENCY FALL DETECTED!")
            st.warning("Notify Caregiver Immediately.")
        else:
            st.success("✅ Normal Activity")

# ==========================================================
# VIDEO DETECTION
# ==========================================================
if mode == "🎥 Video Detection":

    video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if video:

        with open("temp_video.mp4", "wb") as f:
            f.write(video.read())

        cap = cv2.VideoCapture("temp_video.mp4")

        frame_box = st.empty()
        activities = []

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            output, label, confidence = predict_frame(frame)

            activities.append(label)

            cv2.putText(
                output,
                f"{label.upper()}  {confidence:.2%}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            frame_box.image(
                cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        cap.release()

        st.success("🎉 Video Analysis Completed!")

        st.subheader("📊 Activity Summary")

        counts = pd.Series(activities).value_counts()

        st.bar_chart(counts)

        col1, col2 = st.columns(2)

        col1.metric("Frames Analysed", len(activities))
        col2.metric("Fall Frames", activities.count("fall"))

        if activities.count("fall") > 0:
            st.error("🚨 Fall Detected in Video")
        else:
            st.success("✅ No Fall Detected")
