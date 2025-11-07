# --- Basic setup & log cleanup ------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization,
                                     GlobalAveragePooling2D, SpatialDropout2D, Activation)
from tensorflow.keras.optimizers import Adam
# EarlyStopping left in import for reference, but not used
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback

# --- Custom callback for clean progress ---
class CleanProgressCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.steps = 0
        self.max_steps = self.params.get('steps', 1) or 1

    def on_batch_end(self, batch, logs=None):
        self.steps += 1
        progress = self.steps / self.max_steps
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        sys.stdout.write(f'\rEpoch {self.epoch+1}/{self.params.get("epochs", 0)} [{bar}] {self.steps}/{self.max_steps}')
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        print(f"Epoch {epoch+1:2d} | "
              f"Loss: {logs.get('loss', 0):.4f} | "
              f"Acc: {logs.get('accuracy', 0):.4f} | "
              f"Val Loss: {logs.get('val_loss', 0):.4f} | "
              f"Val Acc: {logs.get('val_accuracy', 0):.4f} | "
              f"LR: {logs.get('lr', 0):.2e}")

def bar(txt=None, ch="=", n=60):
    print("\n" + ch * n)
    if txt:
        print(txt)
        print(ch * n)

# ==============================================================================
# 1) DATA: directories, augmentation, generators
# ==============================================================================
bar("STEP 1 • DATA PREP")

# Project requirement: Input shape MUST be (500, 500, 3)
IMG_H, IMG_W, IMG_C = 500, 500, 3
INPUT_SHAPE = (IMG_H, IMG_W, IMG_C)
BATCH_SZ = 32
N_CLASSES = 3

print(f"Input shape: {INPUT_SHAPE}")
print(f"Batch size : {BATCH_SZ}")
print(f"Classes    : {N_CLASSES}")

# Relative folder structure
TRAIN_DIR = "Data/train"
VAL_DIR   = "Data/valid"
TEST_DIR  = "Data/test"

print("\nConfiguring augmentation (aligned with reference settings)...")
# ► Match the reference code's successful augmentation values
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.10,
    zoom_range=0.10,
    rotation_range=10,
    width_shift_range=0.20,
    height_shift_range=0.20,
    horizontal_flip=True,
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(rescale=1.0/255)

print("✓ Augmentation configured")
print("  • Train: rescale + shear/zoom 0.1 + rot 10° + shifts 0.2 + flip")
print("  • Validation: rescale only")

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
print(f"Steps/epoch (train): {max(1, train_gen.n // BATCH_SZ)}")
print(f"Steps/epoch (val)  : {max(1, val_gen.n // BATCH_SZ)}")

# Verify expected counts (per project brief)
expected_train, expected_val = 1942, 431
if train_gen.n == expected_train:
    print(f"✓ Train count matches expected {expected_train}")
else:
    print(f"⚠ Train count {train_gen.n} ≠ expected {expected_train}")

if val_gen.n == expected_val:
    print(f"✓ Val count matches expected {expected_val}")
else:
    print(f"⚠ Val count {val_gen.n} ≠ expected {expected_val}")

print("\n✓ Step 1 complete")

# ==============================================================================
# 2) MODEL: Two-conv blocks + BN→ReLU + He init + light L2 (keeps your structure)
#    Filter plan adjusted to match ref capacity (32→64→128→128→256)
# ==============================================================================
bar("STEP 2 • MODEL")

def conv_block(model, filters, name, l2=1e-4, pool=True, spatial_drop=0.0):
    # Conv → BN → ReLU (BN before ReLU), He init, no bias w/ BN
    model.add(Conv2D(filters, (3, 3), padding="same", use_bias=False,
                     kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(l2), name=f"{name}_conv1"))
    model.add(BatchNormalization(name=f"{name}_bn1"))
    model.add(Activation("relu", name=f"{name}_relu1"))

    model.add(Conv2D(filters, (3, 3), padding="same", use_bias=False,
                     kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(l2), name=f"{name}_conv2"))
    model.add(BatchNormalization(name=f"{name}_bn2"))
    model.add(Activation("relu", name=f"{name}_relu2"))

    if spatial_drop > 0:
        model.add(SpatialDropout2D(spatial_drop, name=f"{name}_spdrop"))
    if pool:
        model.add(MaxPooling2D((2, 2), name=f"{name}_pool"))

model = Sequential(name="AircraftDefectCNN_plus")
model.add(layers.Input(shape=INPUT_SHAPE))

print("Assembling CNN: two-conv blocks, BN→ReLU, He init, GAP head, light L2")

# Capacity mirrors the 60% ref (but with stronger per-block convs)
conv_block(model,  32, "b1", l2=1e-4, pool=True,  spatial_drop=0.00)
conv_block(model,  64, "b2", l2=1e-4, pool=True,  spatial_drop=0.00)
conv_block(model, 128, "b3", l2=1e-4, pool=True,  spatial_drop=0.05)
conv_block(model, 128, "b4", l2=1e-4, pool=True,  spatial_drop=0.05)
conv_block(model, 256, "b5", l2=1e-4, pool=True,  spatial_drop=0.10)

# GAP head (safer than Flatten at 500×500)
model.add(GlobalAveragePooling2D(name="global_pool"))

# Dense head: Dense → BN → ReLU (no bias) + Dropout
model.add(Dense(256, use_bias=False, kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4), name="dense1"))
model.add(BatchNormalization(name="dense1_bn"))
model.add(Activation("relu", name="dense1_relu"))
model.add(Dropout(0.30))  # keep 0.30 to avoid underfitting

model.add(Dense(128, use_bias=False, kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4), name="dense2"))
model.add(BatchNormalization(name="dense2_bn"))
model.add(Activation("relu", name="dense2_relu"))
model.add(Dropout(0.30))

model.add(Dense(N_CLASSES, activation="softmax", name="out"))

bar("MODEL SUMMARY")
model.summary()
print("\n✓ Step 2 complete")

# ==============================================================================
# 3) COMPILE: Use ref LR (5e-4) + clipnorm; optional label smoothing (set to 0.0 to match ref)
# ==============================================================================
bar("STEP 3 • COMPILE & HYPERPARAMS")

print("Compiling with categorical_crossentropy / Adam(5e-4) / accuracy")
opt = Adam(learning_rate=5e-4, clipnorm=1.0)

# If you prefer pure parity with the ref, set label_smoothing=0.0
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

model.compile(loss=loss_fn, optimizer=opt, metrics=["accuracy"])

print("\nHyperparameters")
print("• Filters: 32 → 64 → 128 → 128 → 256")
print("• Two 3×3 convs per block, BN→ReLU, He init, L2(1e-4)")
print("• GAP head; Dense 256→128→softmax(3), Dropout 0.30")
print(f"• Image size: {IMG_H}×{IMG_W}×{IMG_C}, Batch size: {BATCH_SZ}")
print("• Learning rate: 5e-4, clipnorm: 1.0")
print("• Label smoothing: 0.05 (set to 0.0 if you want exact ref behavior)")
print("\n✓ Step 3 complete")

# ==============================================================================
# 4) TRAIN: run full 30 epochs (EarlyStopping DISABLED)
# ==============================================================================
bar("STEP 4 • TRAIN & EVALUATE")

EPOCHS = 30  # always run full 30

print("Setting callbacks...")
cbs = [
    # EarlyStopping is intentionally disabled so we don't stop before 30 epochs:
    # EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    CleanProgressCallback()
]
print("✓ EarlyStopping disabled (commented out) → training will run full 30 epochs")
print("✓ ReduceLROnPlateau(factor=0.5, pat=4, min_lr=1e-7)")
print("✓ CleanProgressCallback")

print("\nTraining configuration")
tr_steps = max(1, train_gen.n // BATCH_SZ)
va_steps = max(1, val_gen.n // BATCH_SZ)
print(f"• Epochs (max): {EPOCHS}")
print(f"• Steps/epoch  : {tr_steps}")
print(f"• Val steps    : {va_steps}")

# Optional: class weights only if imbalance is meaningful
print("\nComputing optional class weights…")
try:
    cls_counts = np.bincount(train_gen.classes, minlength=N_CLASSES).astype(np.float32)
    ratio = float(cls_counts.max() / max(1.0, cls_counts.min()))
    if ratio < 1.5:
        cls_weights = None
        print(f"Class counts: {cls_counts.tolist()} | Imbalance ratio={ratio:.2f} → not using class_weight")
    else:
        cls_weights = {i: (train_gen.n / (N_CLASSES * cls_counts[i])) if cls_counts[i] > 0 else 1.0
                       for i in range(N_CLASSES)}
        print(f"Class counts: {cls_counts.tolist()} | Using class_weight: {cls_weights}")
except Exception as e:
    print(f"Could not compute class weights: {e}")
    cls_weights = None

bar("TRAINING")
t0 = time.time()
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=cbs,
    verbose=0,
    class_weight=cls_weights  # safe to be None
)

t1 = time.time()
elapsed = t1 - t0
hh, mm, ss = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
avg_per_epoch = elapsed / max(1, len(history.history.get("loss", [])) or 1)

bar("TRAINING COMPLETE")
print(f"End   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Time  : {hh}h {mm}m {ss}s  ({elapsed:.2f}s total)")
print(f"Epochs: {len(history.history.get('loss', []))} of {EPOCHS}")
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
plt.plot(epochs_range, history.history["accuracy"], label="Train Acc", linewidth=2, marker="o", markersize=6)
plt.plot(epochs_range, history.history["val_accuracy"], label="Val Acc", linewidth=2, marker="s", markersize=6)
plt.title("Accuracy", fontsize=14, fontweight="bold")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.ylim([0, 1])
plt.xlim([0.5, len(history.history["accuracy"]) + 0.5]); plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history["loss"], label="Train Loss", linewidth=2, marker="o", markersize=6)
plt.plot(epochs_range, history.history["val_loss"], label="Val Loss", linewidth=2, marker="s", markersize=6)
plt.title("Loss", fontsize=14, fontweight="bold")
plt.xlabel("Epoch"); plt.ylabel("Loss")
max_loss = max(max(history.history["loss"]), max(history.history["val_loss"]))
plt.ylim([0, min(max_loss * 1.1, 8)]); plt.xlim([0.5, len(history.history["loss"]) + 0.5])
plt.legend(loc="upper right"); plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = "outputs/model_performance.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved curves → {plot_file}")

# final metrics
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
    print("\n⚠ Some overfitting detected")
elif va_acc > tr_acc:
    print("\n✓ Excellent: validation ≥ training!")
else:
    print(f"\n✓ Reasonable generalization (gap {gap*100:.2f}%)")

bar("PIPELINE DONE")
print("Final settings:")
print("  • EarlyStopping DISABLED (commented) → always runs 30 epochs")
print("  • RLROP active to lower LR on plateaus")
print("  • Stronger blocks: BN→ReLU, He init, L2(1e-4), GAP head")
print("  • Aug: shear/zoom=0.1, rot=10°, shifts=0.2, flip")
print("\nArtifacts:")
print(f"  • {model_path}")
print(f"  • {plot_file}")
