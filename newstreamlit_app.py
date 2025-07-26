
import streamlit as st
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import cv2
from datetime import datetime
import os

st.set_page_config(page_title="Face Attendance via Webcam", layout="centered")
st.title("📸 Face Recognition Attendance System")
st.markdown("Upload or capture a photo and mark your attendance.")

# Load trained data
with open("data/faces_data.pkl", "rb") as f:
    faces_data = pickle.load(f)
with open("data/names.pkl", "rb") as f:
    names = pickle.load(f)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces_data, names)

# Attendance file
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Date,Time\n")

# Webcam snapshot or upload
img_file = st.camera_input("Take a picture")
if not img_file:
    img_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

if img_file:
    img = Image.open(img_file).convert('RGB')
    img_array = np.array(img)

    # Convert to grayscale and detect face
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected.")
    else:
        for (x, y, w, h) in faces:
            roi = img_array[y:y+h, x:x+w]
            roi = cv2.resize(roi, (50, 50)).flatten().reshape(1, -1)
            pred_name = knn.predict(roi)[0]

            # Mark attendance if not already
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            with open(attendance_file, "r") as f:
                already_marked = f.read()

            if pred_name not in already_marked:
                with open(attendance_file, "a") as f:
                    f.write(f"{pred_name},{date_str},{time_str}\n")

            st.success(f"✅ Attendance marked for: {pred_name}")

            # Show marked image
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_array, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        st.image(img_array, caption="Detected Face(s)", use_column_width=True)
