# --- Basic setup & log cleanup ------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # silence TF info/warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # avoid oneDNN quirks on some CPUs
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"  # hush grappler chatter

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# prefer legacy Adam on Apple silicon installs
try:
    from tensorflow.keras.optimizers.legacy import Adam
except ImportError:
    from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Small helper for tidy console banners -----------------------------------
def bar(txt=None, ch="=", n=60):
    print("\n" + ch * n)
    if txt:
        print(txt)
        print(ch * n)

# ==============================================================================
# 1) DATA: directories, augmentation, generators
# ==============================================================================
bar("STEP 1 • DATA PREP")

# image / loader config
IMG_H, IMG_W, IMG_C = 500, 500, 3
INPUT_SHAPE = (IMG_H, IMG_W, IMG_C)
BATCH_SZ = 32
N_CLASSES = 3

print(f"Input shape: {INPUT_SHAPE}")
print(f"Batch size : {BATCH_SZ}")
print(f"Classes    : {N_CLASSES}")

# relative folder structure (train/valid/test with class subfolders)
TRAIN_DIR = "Data/train"
VAL_DIR   = "Data/valid"
TEST_DIR  = "Data/test"

print("\nConfiguring augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=[0.85, 1.15],
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

print("✓ Augmentation ready (train augmented, validation rescaled only)")

print("\nBuilding generators...")
try:
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_H, IMG_W),
        batch_size=BATCH_SZ,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    print(f"✓ Train: {train_gen.n} files")
except Exception as e:
    print(f"Training generator error: {e}")
    raise SystemExit(1)

try:
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_H, IMG_W),
        batch_size=BATCH_SZ,
        class_mode="categorical",
        shuffle=False,
        seed=42,
    )
    print(f"✓ Valid: {val_gen.n} files")
except Exception as e:
    print(f"Validation generator error: {e}")
    raise SystemExit(1)

bar("DATA SUMMARY")
print(f"Train samples     : {train_gen.n}")
print(f"Validation samples: {val_gen.n}")
print(f"Class map         : {train_gen.class_indices}")
print(f"Steps/epoch (train): {train_gen.n // BATCH_SZ}")
print(f"Steps/epoch (val)  : {val_gen.n // BATCH_SZ}")

expected_train, expected_val = 1942, 431
if train_gen.n != expected_train:
    print(f"⚠ Train count {train_gen.n} ≠ expected {expected_train} (OK if split differs)")
else:
    print(f"✓ Train count matches expected {expected_train}")

if val_gen.n != expected_val:
    print(f"⚠ Val count {val_gen.n} ≠ expected {expected_val} (OK if split differs)")
else:
    print(f"✓ Val count matches expected {expected_val}")

print("\n✓ Step 1 complete")

# ==============================================================================
# 2) MODEL: CNN architecture
# ==============================================================================
bar("STEP 2 • MODEL")

model = Sequential(name="AircraftDefectCNN")

print("Assembling CNN...")
# Block 1
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE, name="conv1"))
model.add(MaxPooling2D((2, 2), name="pool1"))

# Block 2
model.add(Conv2D(64, (3, 3), activation="relu", name="conv2"))
model.add(MaxPooling2D((2, 2), name="pool2"))

# Block 3
model.add(Conv2D(128, (3, 3), activation="relu", name="conv3"))
model.add(MaxPooling2D((2, 2), name="pool3"))

# Block 4
model.add(Conv2D(128, (3, 3), activation="relu", name="conv4"))
model.add(MaxPooling2D((2, 2), name="pool4"))

# Block 5
model.add(Conv2D(256, (3, 3), activation="relu", name="conv5"))
model.add(MaxPooling2D((2, 2), name="pool5"))

# Dense head
model.add(Flatten(name="flatten"))
model.add(Dense(256, activation="relu", name="dense1"))
model.add(Dropout(0.5, name="dropout1"))
model.add(Dense(128, activation="relu", name="dense2"))
model.add(Dropout(0.5, name="dropout2"))
model.add(Dense(N_CLASSES, activation="softmax", name="out"))

bar("MODEL SUMMARY")
model.summary()
print("\n✓ Step 2 complete")

# ==============================================================================
# 3) COMPILE: loss/optimizer/metrics
# ==============================================================================
bar("STEP 3 • COMPILE & HYPERPARAMS")

print("Compiling with categorical_crossentropy / Adam(lr=5e-4) / accuracy")
opt = Adam(learning_rate=5e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("\nHyperparameters")
print("• Convs: filters 32→64→128→128→256, kernel 3×3, ReLU, MaxPool 2×2")
print("• Dense: 256→128→softmax(3), Dropout 0.5")
print(f"• Image size: {IMG_H}×{IMG_W}×{IMG_C}")
print(f"• Batch size: {BATCH_SZ}")

print("\n✓ Step 3 complete")

# ==============================================================================
# 4) TRAIN: callbacks, fit, save, plots, quick diagnostics
# ==============================================================================
bar("STEP 4 • TRAIN & EVALUATE")

EPOCHS = 30  # long, early stopping will intervene if needed

print("Setting callbacks...")
cbs = [
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
]
print("✓ EarlyStopping(patience=8), ReduceLROnPlateau(factor=0.5, patience=4)")

print("\nTraining configuration")
tr_steps = train_gen.n // BATCH_SZ
va_steps = val_gen.n // BATCH_SZ
print(f"• Epochs (max): {EPOCHS}")
print(f"• Steps/epoch  : {tr_steps}  (~{tr_steps * BATCH_SZ} images/epoch)")
print(f"• Val steps    : {va_steps}")

bar("TRAINING")
t0 = time.time()
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=cbs,
    verbose=1,
)

t1 = time.time()
elapsed = t1 - t0
hh, mm, ss = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
avg_per_epoch = elapsed / max(1, len(history.history["loss"]))

bar("TRAINING COMPLETE")
print(f"End   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Time  : {hh}h {mm}m {ss}s  ({elapsed:.2f}s total)")
print(f"Epochs: {len(history.history['loss'])} of {EPOCHS}")
print(f"Avg/epoch: {avg_per_epoch:.2f}s")

# save model
os.makedirs("models", exist_ok=True)
model_path = "models/aircraft_defect_model.keras"
model.save(model_path)
print(f"\n✓ Saved model → {model_path}")

# performance curves
bar("PLOTS")
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
epochs_range = range(1, len(history.history["accuracy"]) + 1)
plt.plot(epochs_range, history.history["accuracy"], label="Train Acc", linewidth=2, marker="o", markersize=8)
plt.plot(epochs_range, history.history["val_accuracy"], label="Val Acc", linewidth=2, marker="s", markersize=8)
plt.title("Accuracy", fontsize=14, fontweight="bold")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.ylim([0, 1])
plt.xlim([0.5, len(history.history["accuracy"]) + 0.5]); plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history["loss"], label="Train Loss", linewidth=2, marker="o", markersize=8)
plt.plot(epochs_range, history.history["val_loss"], label="Val Loss", linewidth=2, marker="s", markersize=8)
plt.title("Loss", fontsize=14, fontweight="bold")
plt.xlabel("Epoch"); plt.ylabel("Loss")
max_loss = max(max(history.history["loss"]), max(history.history["val_loss"]))
plt.ylim([0, max_loss * 1.1]); plt.xlim([0.5, len(history.history["loss"]) + 0.5])
plt.legend(loc="upper right"); plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = "outputs/model_performance.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved curves → {plot_file}")

# final metrics + basic overfit signal
bar("FINAL METRICS")
tr_acc = history.history["accuracy"][-1]
va_acc = history.history["val_accuracy"][-1]
tr_loss = history.history["loss"][-1]
va_loss = history.history["val_loss"][-1]

print(f"Train Acc: {tr_acc:.4f}  ({tr_acc*100:.2f}%)")
print(f"Val   Acc: {va_acc:.4f}  ({va_acc*100:.2f}%)")
print(f"Train Loss: {tr_loss:.4f}")
print(f"Val   Loss: {va_loss:.4f}")

gap = tr_acc - va_acc
if gap > 0.15:
    print("\n⚠ Likely overfitting:")
    print("  • Consider higher dropout, stronger aug, or a smaller head.")
elif va_acc > tr_acc:
    print("\n✓ Nice: validation ≥ training (dropout often causes this).")
else:
    print(f"\n✓ Looks reasonable (gap {gap*100:.2f}%).")

bar("PIPELINE DONE")
print("Next up:")
print("  1) Inspect outputs/model_performance.png")
print("  2) If needed, tweak aug/dropout/filters and retrain")
print("  3) Use your Step-5 script to run predictions on held-out images")
print("\nArtifacts:")
print(f"  • {model_path}")
print(f"  • {plot_file}")
