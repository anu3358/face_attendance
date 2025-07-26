import cv2
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load Haar Cascade for face detection
facedetect = cv2.CascadeClassifier('C:/Users/dell/OneDrive/Desktop/coding__/PYTHON/DATA/haarcascade_frontalface_default.xml')

# Load the stored face data and names
with open('data/names.pkl', 'rb') as f:
    names = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    faces_data = pickle.load(f)

# Fix any length mismatch
if len(faces_data) != len(names):
    print(f"[WARNING] Length mismatch: faces={len(faces_data)}, names={len(names)}")
    min_len = min(len(faces_data), len(names))
    faces_data = faces_data[:min_len]
    names = names[:min_len]
    print(f"[INFO] Trimmed both lists to {min_len} samples.")

# Train KNN classifier
print("[INFO] Training classifier...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces_data, names)
print("[INFO] Classifier trained. Starting recognition...")

# Start webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # Predict name using KNN
        prediction = knn.predict(resized_img)
        name = prediction[0]

        # Display name
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
