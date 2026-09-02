import streamlit as st
import cv2, joblib, numpy as np, pandas as pd
import tensorflow as tf
from ultralytics import YOLO

st.set_page_config(page_title="SafeFall AI", page_icon="🚨", layout="wide")

pose_model = YOLO("yolov8n-pose.pt")
classifier = tf.keras.models.load_model("model/activity_classifier.keras")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/label_encoder.pkl")

def predict_frame(frame):
    result = pose_model(frame, verbose=False)
    annotated = result[0].plot()
    kp = result[0].keypoints.xy
    if kp is None or len(kp)==0:
        return annotated, "No Person", 0
    feat = scaler.transform([kp[0].cpu().numpy().flatten()])
    pred = classifier.predict(feat, verbose=False)[0]
    label = encoder.inverse_transform([np.argmax(pred)])[0]
    conf = float(np.max(pred))
    return annotated, label, conf

st.title("🚨 SafeFall AI - Elderly Fall Detection")

choice = st.sidebar.radio("Input Type", ["Image Detection","Video Detection"])

if choice=="Image Detection":
    file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    if file:
        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8),1)
        annotated,label,conf = predict_frame(frame)
        st.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB), use_container_width=True)
        st.metric("Prediction",label)
        st.metric("Confidence",f"{conf:.2%}")
        if label=="fall" and conf>0.6:
            st.error("🚨 Emergency Fall Detected! Notify Caregiver Immediately.")
        else:
            st.success("Normal Activity")

if choice=="Video Detection":
    video = st.file_uploader("Upload Video", type=["mp4","avi"])
    if video:
        open("temp_video.mp4","wb").write(video.read())
        cap = cv2.VideoCapture("temp_video.mp4")
        frame_box = st.empty()
        activities=[]
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            annotated,label,conf = predict_frame(frame)
            activities.append(label)
            frame_box.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB), use_container_width=True)
        cap.release()
        st.success("Video Analysis Complete")
        st.bar_chart(pd.Series(activities).value_counts())
