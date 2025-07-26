
import streamlit as st
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from datetime import datetime
import os
import face_recognition

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

    # Use face_recognition to locate faces and get encodings
    face_locations = face_recognition.face_locations(img_array)
    face_encodings = face_recognition.face_encodings(img_array, face_locations)

    if not face_encodings:
        st.warning("No face detected.")
    else:
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            face_flat = face_encoding.reshape(1, -1)  # Already a 128-d feature vector
            pred_name = knn.predict(face_flat)[0]

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

            # Annotate image
            annotated_img = img_array.copy()
            annotated_img[top:bottom, left:right] = np.clip(annotated_img[top:bottom, left:right], 0, 255)
            st.image(annotated_img, caption=f"Detected: {pred_name}", use_column_width=True)
