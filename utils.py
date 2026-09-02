import cv2
import numpy as np
from ultralytics import YOLO

pose_model = YOLO("yolov8n-pose.pt")

def extract_landmarks_from_frame(frame):
    results = pose_model(frame, verbose=False)
    if len(results)==0 or results[0].keypoints.xy is None:
        return np.zeros(34)
    return results[0].keypoints.xy[0].cpu().numpy().flatten()

def draw_landmarks_on_frame(frame):
    return pose_model(frame, verbose=False)[0].plot()
