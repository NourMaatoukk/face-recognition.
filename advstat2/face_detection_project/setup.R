# quick_setup.R

library(reticulate)
library(fs)

# ===========================
image_dir <- "c:/Users/Joe/OneDrive/Desktop/New folder/face_detection_project/images/images"
output_file <- "c:/Users/Joe/OneDrive/Desktop/New folder/face_detection_project/face_encodings.pkl"

if (!dir_exists(image_dir)) {
  stop("Cannot find image file directory: ", image_dir)
}

# استخدم البيئة الافتراضية الخاصة بـ conda
use_condaenv("face310", required = TRUE)

# كود Python كامل صحيح
py_run_string("
import os
import torch
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

IMAGES_FOLDER = r'C:/Users/Joe/OneDrive/Desktop/New folder/face_detection_project/images/images'
OUTPUT_PATH = r'C:/Users/Joe/OneDrive/Desktop/New folder/face_detection_project/face_encodings.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=False, min_face_size=60, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeddings = []
names = []

print('\\n🚀 Starting face embedding training...\\n')

for filename in os.listdir(IMAGES_FOLDER):
    if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
        continue

    img_path = os.path.join(IMAGES_FOLDER, filename)
    try:
        img = Image.open(img_path).convert('RGB')
    except:
        print(f'Skipping corrupted file: {filename}')
        continue

    face = mtcnn(img)
    if face is None:
        print(f'❌ No face detected in: {filename}')
        continue

    if len(face.shape) == 3:
        face = face.unsqueeze(0)
    face = face.to(device)

    with torch.no_grad():
        emb = resnet(face).cpu().numpy().flatten()

    emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
    clean_name = filename.split('_')[0]

    embeddings.append(emb_norm)
    names.append(clean_name)

    print(f'✔️ Processed: {filename}  → Name: {clean_name}')

data = {'embeddings': np.array(embeddings), 'names': names}

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(data, f)

print(f'\\n✅ All embeddings saved successfully! Total faces: {len(names)}')
")

cat("🎉 Quick setup completed!\n")
