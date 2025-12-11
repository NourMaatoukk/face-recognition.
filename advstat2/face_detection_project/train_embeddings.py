import os
import torch
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Path to your images folder
IMAGES_FOLDER = r"C:\Users\Joe\OneDrive\Desktop\New folder\face_detection_project\images\images"
OUTPUT_PATH = r"C:\Users\Joe\OneDrive\Desktop\New folder\face_detection_project\face_encodings.pkl"

# Load face detector + facenet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SAME SETTINGS AS face_recognition.py
mtcnn = MTCNN(keep_all=False, min_face_size=60, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeddings = []
names = []

print("\n🚀 Starting face embedding training...\n")

for filename in os.listdir(IMAGES_FOLDER):
    if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
        continue

    img_path = os.path.join(IMAGES_FOLDER, filename)

    try:
        img = Image.open(img_path).convert('RGB')
    except:
        print(f"Skipping corrupted file: {filename}")
        continue

    # Detect face
    face = mtcnn(img)
    if face is None:
        print(f"❌ No face detected in: {filename}")
        continue

    # Prepare face tensor
    if len(face.shape) == 3:
        face = face.unsqueeze(0)
    face = face.to(device)

    # Extract embedding
    with torch.no_grad():
        emb = resnet(face).cpu().numpy().flatten()

    # Normalize embedding (VERY IMPORTANT)
    emb_norm = emb / (np.linalg.norm(emb) + 1e-10)

    # Extract clean name (take before first underscore)
    clean_name = filename.split("_")[0]

    embeddings.append(emb_norm)
    names.append(clean_name)

    print(f"✔️ Processed: {filename}  → Name: {clean_name}")

# Save embeddings
data = {"embeddings": np.array(embeddings), "names": names}

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(data, f)

print("\n✅ All embeddings saved successfully!")
print(f"🧑‍🎓 Total faces saved: {len(names)}")
print(f"📁 Output file: {OUTPUT_PATH}")
