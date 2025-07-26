import streamlit as st
import cv2
import face_recognition
import numpy as np
import pickle
import os
from datetime import datetime

# Load known faces and names
with open("data/faces_data.pkl", "rb") as f:
    known_face_encodings = pickle.load(f)
with open("data/names.pkl", "rb") as f:
    known_face_names = pickle.load(f)

# Ensure attendance file exists
if not os.path.exists("attendance.csv"):
    with open("attendance.csv", "w") as f:
        f.write("Name,Time\n")

def mark_attendance(name):
    with open("attendance.csv", "r+") as f:
        lines = f.readlines()
        recorded_names = [line.split(',')[0] for line in lines]
        if name not in recorded_names:
            now = datetime.now()
            time_str = now.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{name},{time_str}\n")

def main():
    st.set_page_config(page_title="Face Attendance System", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 Face Recognition Attendance System</h1>", unsafe_allow_html=True)

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to access webcam.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                mark_attendance(name)

                # Draw bounding box and label
                top, right, bottom, left = [v * 4 for v in face_location]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Stop camera with keyboard interrupt
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    st.markdown("---")
    st.subheader("📋 Today's Attendance")
    with open("attendance.csv", "r") as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            name, time = line.strip().split(',')
            st.write(f"✅ {name} - {time}")

if __name__ == '__main__':
    main()
