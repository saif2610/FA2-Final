# SafeFall AI Training (Google Colab)
!pip install ultralytics opencv-python tensorflow scikit-learn joblib matplotlib

import os, cv2, joblib, numpy as np
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

pose_model = YOLO("yolov8n-pose.pt")

DATASET="/content/Le2i_Dataset"
CLASSES=["fall","walking","sitting","standing","normal"]

def extract_pose(frame):
    result = pose_model(frame, verbose=False)
    if len(result)==0 or result[0].keypoints.xy is None:
        return np.zeros(34)
    return result[0].keypoints.xy[0].cpu().numpy().flatten()

features,labels=[],[]

for cls in CLASSES:
    folder=os.path.join(DATASET,cls)
    for img in os.listdir(folder):
        frame=cv2.imread(os.path.join(folder,img))
        if frame is None: continue
        frame=cv2.resize(frame,(640,640))
        features.append(extract_pose(frame))
        labels.append(cls)

X=np.array(features)
encoder=LabelEncoder()
y=encoder.fit_transform(labels)

scaler=StandardScaler()
X=scaler.fit_transform(X)

joblib.dump(scaler,"scaler.pkl")
joblib.dump(encoder,"label_encoder.pkl")

X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.50,random_state=42,stratify=y_temp)

model=Sequential([
    Dense(128,activation="relu",input_shape=(34,)),
    Dropout(0.3),
    Dense(64,activation="relu"),
    Dropout(0.3),
    Dense(32,activation="relu"),
    Dense(len(CLASSES),activation="softmax")
])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

history=model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=25,batch_size=32)

model.save("activity_classifier.keras")
print("Training Complete.")
