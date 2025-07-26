import os
import pickle

def check_file(path):
    if os.path.exists(path):
        print(f"[✓] {path} exists")
        return True
    else:
        print(f"[✗] {path} is missing")
        return False

def check_pickle_lengths(faces_path, names_path):
    with open(faces_path, 'rb') as f:
        faces = pickle.load(f)
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    if len(faces) == len(names):
        print(f"[✓] faces_data and names length match ({len(faces)})")
    else:
        print(f"[✗] faces_data ({len(faces)}) and names ({len(names)}) length mismatch")

if __name__ == "__main__":
    print("🔍 Running Project Tests...\n")
    check_file('add_faces.py')
    check_file('app.py')
    check_file('attendance.csv')
    check_file('data/faces_data.pkl')
    check_file('data/names.pkl')
    check_file('requirements.txt')
    
    if check_file('data/faces_data.pkl') and check_file('data/names.pkl'):
        check_pickle_lengths('data/faces_data.pkl', 'data/names.pkl')
