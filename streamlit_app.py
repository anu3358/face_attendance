
import streamlit as st
from deepface import DeepFace
import numpy as np
from PIL import Image
import os
from datetime import datetime
import pandas as pd
import base64

st.set_page_config(page_title="Face Attendance with DeepFace", layout="centered")
st.title("🧠 Face Attendance System with DeepFace")
st.markdown("Use webcam or upload photo to mark your attendance.")

# Attendance CSV
ATTENDANCE_FILE = "attendance.csv"
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

# Upload reference images and names
REFERENCE_DIR = "reference_images"
os.makedirs(REFERENCE_DIR, exist_ok=True)

# Load known people from reference folder
known_faces = []
known_names = []
for file in os.listdir(REFERENCE_DIR):
    if file.endswith(('.png', '.jpg', '.jpeg')):
        known_faces.append(os.path.join(REFERENCE_DIR, file))
        known_names.append(os.path.splitext(file)[0])

# Upload new reference images
with st.expander("🧑‍💼 Add New Person"):
    ref_name = st.text_input("Enter Name")
    ref_img = st.file_uploader("Upload a Clear Face Image", type=["jpg", "jpeg", "png"])
    if st.button("Save Reference"):
        if ref_name and ref_img:
            save_path = os.path.join(REFERENCE_DIR, f"{ref_name}.jpg")
            with open(save_path, "wb") as f:
                f.write(ref_img.getbuffer())
            st.success(f"Saved {ref_name} to reference_images.")
            st.experimental_rerun()
        else:
            st.warning("Please enter a name and upload an image.")

# Camera or upload input
img_file = st.camera_input("📷 Take a picture")
if not img_file:
    img_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img.save("temp.jpg")

    try:
        result = DeepFace.find(img_path="temp.jpg", db_path=REFERENCE_DIR, enforce_detection=False, model_name="Facenet")
        if not result.empty:
            best_match = result.iloc[0]
            pred_name = os.path.splitext(os.path.basename(best_match["identity"]))[0]

            # Attendance mark
            df = pd.read_csv(ATTENDANCE_FILE)
            today = datetime.now().strftime("%Y-%m-%d")
            if not ((df["Name"] == pred_name) & (df["Date"] == today)).any():
                new_entry = {"Name": pred_name, "Date": today, "Time": datetime.now().strftime("%H:%M:%S")}
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                df.to_csv(ATTENDANCE_FILE, index=False)
                st.success(f"✅ Attendance marked for {pred_name}")
            else:
                st.info(f"{pred_name}'s attendance is already marked today.")
        else:
            st.warning("No match found.")

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Show current attendance
tab1, tab2 = st.tabs(["📄 Attendance Log", "📁 Reference Images"])

with tab1:
    df = pd.read_csv(ATTENDANCE_FILE)
    st.dataframe(df)

with tab2:
    for name, face_path in zip(known_names, known_faces):
        st.image(face_path, caption=name, width=120)

# Cleanup temp
if os.path.exists("temp.jpg"):
    os.remove("temp.jpg")
