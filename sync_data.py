import pickle

with open('data/names.pkl', 'rb') as f:
    names = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    faces = pickle.load(f)

min_len = min(len(faces), len(names))

names = names[:min_len]
faces = faces[:min_len]

with open('data/names.pkl', 'wb') as f:
    pickle.dump(names, f)

with open('data/faces_data.pkl', 'wb') as f:
    pickle.dump(faces, f)

print(f"[✓] Synced data length to {min_len}")
