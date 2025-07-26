import cv2
import pickle
import numpy as np
import os
import csv
from datetime import datetime
import pyttsx3
from sklearn.neighbors import KNeighborsClassifier

# Voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Haar Cascade
facedetect = cv2.CascadeClassifier('C:/Users/dell/OneDrive/Desktop/coding__/PYTHON/DATA/haarcascade_frontalface_default.xml')

# Load data
with open('data/names.pkl', 'rb') as f:
    names = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    faces_data = pickle.load(f)

# Sync names/faces if length mismatch
if len(faces_data) != len(names):
    min_len = min(len(faces_data), len(names))
    faces_data = faces_data[:min_len]
    names = names[:min_len]

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces_data, names)

# Attendance file setup
attendance_file = 'attendance.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Date', 'Time'])

marked_names = set()

# Start webcam
video = cv2.VideoCapture(0)

print("[INFO] Starting Face Recognition...")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        prediction = knn.predict(resized_img)
        name = prediction[0]

        # Draw info
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Attendance logging
        if name not in marked_names:
            now = datetime.now()
            date = now.strftime('%Y-%m-%d')
            time = now.strftime('%H:%M:%S')
            with open(attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, date, time])
            marked_names.add(name)
            engine.say(f"Welcome {name}")
            engine.runAndWait()

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
