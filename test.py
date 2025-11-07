import os, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# --- Config (match training) ---
IMG_H, IMG_W = 500, 500
MODEL_PATH   = "models/bad_aircraft_defect_model.keras"
TRAIN_DIR    = "Data/train"   # fallback to rebuild class order if JSON missing
TEST_DIR     = "Data/test"

# --- Load model ---
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# --- Load class mapping (prefer JSON from training; else rebuild from TRAIN_DIR) ---
idx_to_class = None
try:
    with open("models/class_indices.json", "r") as f:
        class_indices = json.load(f)                 # e.g., {"crack":0,"missing-head":1,"paint-off":2}
    idx_to_class = {v: k for k, v in class_indices.items()}
except Exception:
    classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    idx_to_class = {i: c for i, c in enumerate(classes)}

def preprocess(path):
    img = load_img(path, target_size=(IMG_H, IMG_W))
    x = img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0), img

def predict_one(path):
    x, pil_img = preprocess(path)
    probs = model.predict(x, verbose=0)[0]
    top = int(np.argmax(probs))
    label = idx_to_class[top]
    conf  = float(probs[top])
    return label, conf, probs, pil_img

# Required three images (per brief)
paths = [
    os.path.join(TEST_DIR, "crack",        "test_crack.jpg"),
    os.path.join(TEST_DIR, "missing-head", "test_missinghead.jpg"),
    os.path.join(TEST_DIR, "paint-off",    "test_paintoff.jpg"),
]

# --- Make a Figure 3–style panel ---
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(12, 4))
for i, p in enumerate(paths, 1):
    label, conf, probs, pil_img = predict_one(p)
    plt.subplot(1, 3, i)
    plt.imshow(pil_img)
    plt.axis("off")
    plt.title(f"{os.path.basename(p)}\nPred: {label} ({conf:.2%})")
plt.tight_layout()
out_path = "outputs/bad_test_predictions.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"✓ Saved test panel → {out_path}")
