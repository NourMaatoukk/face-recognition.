import pickle

with open("face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys()) 
