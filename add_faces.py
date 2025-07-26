import cv2
import pickle
import numpy as np
import os

# Show current working directory
print("Current Working Directory:", os.getcwd())

# Ensure 'data/' directory exists
if not os.path.exists('data'):
    os.makedirs('data')
    print("[INFO] Created 'data' folder.")

# Load Haar Cascade for face detection
facedetect = cv2.CascadeClassifier('C:/Users/dell/OneDrive/Desktop/coding__/PYTHON/DATA/haarcascade_frontalface_default.xml')

# Start webcam
video = cv2.VideoCapture(0)

faces_data = []
i = 0

name = input("Enter Your Name: ")

print("[INFO] Starting face capture. Press 'q' to quit early.")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))

        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)

        i += 1

        cv2.putText(frame, f"Images: {len(faces_data)}", (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# Convert face data to NumPy array and reshape
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)  # Each image is flattened (50x50x3 = 7500)

# File paths
names_path = 'data/names.pkl'
faces_path = 'data/faces_data.pkl'

# Save or update names
if not os.path.exists(names_path):
    names = [name] * 100
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)
    print("[INFO] names.pkl created.")
else:
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)
    print("[INFO] names.pkl updated.")

# Save or update faces data
if not os.path.exists(faces_path):
    with open(faces_path, 'wb') as f:
        pickle.dump(faces_data, f)
    print("[INFO] faces_data.pkl created.")
else:
    with open(faces_path, 'rb') as f:
        existing_faces = pickle.load(f)

    if existing_faces.shape[1] != faces_data.shape[1]:
        print("[WARNING] Shape mismatch detected. Resetting faces_data.pkl.")
        os.remove(faces_path)
        with open(faces_path, 'wb') as f:
            pickle.dump(faces_data, f)
        print("[INFO] faces_data.pkl reset and saved.")
    else:
        faces = np.append(existing_faces, faces_data, axis=0)
        with open(faces_path, 'wb') as f:
            pickle.dump(faces, f)
        print("[INFO] faces_data.pkl updated.")
