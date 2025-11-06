"""
Aircraft Surface Defect Classification - CNN Training Script
------------------------------------------------------------
Author: [Your Name]
Description:
    - Loads and preprocesses aircraft surface images
    - Builds and trains a CNN to detect defect types:
        1. Crack
        2. Missing Head
        3. Paint Off
    - Saves trained model and training performance plots
"""

# ============================================================
# Imports and Environment Setup
# ============================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ============================================================
# STEP 1: DATA PREPARATION
# ============================================================

print("\n" + "="*60)
print("STEP 1: DATA PREPARATION")
print("="*60)

# Image parameters
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 500, 500, 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
BATCH_SIZE = 32
NUM_CLASSES = 3

print(f"Image shape: {IMG_SHAPE}, Batch size: {BATCH_SIZE}, Classes: {NUM_CLASSES}")

# Directory paths
TRAIN_DIR = 'Data/train'
VALID_DIR = 'Data/valid'
TEST_DIR = 'Data/test'

# Data augmentation for training
train_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest'
)

# Only rescaling for validation
valid_gen = ImageDataGenerator(rescale=1./255)

# Data generators
print("\nCreating image generators...")
train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

valid_data = valid_gen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

print(f"\nTrain samples: {train_data.n}, Validation samples: {valid_data.n}")
print(f"Classes: {train_data.class_indices}")

# ============================================================
# STEP 2: MODEL ARCHITECTURE
# ============================================================

print("\n" + "="*60)
print("STEP 2: MODEL ARCHITECTURE")
print("="*60)

model = Sequential(name='Aircraft_Defect_CNN')

# --- Convolutional blocks ---
model.add(Conv2D(32, (3,3), activation='relu', input_shape=IMG_SHAPE))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

# --- Dense layers ---
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.summary()

# ============================================================
# STEP 3: COMPILATION
# ============================================================

print("\n" + "="*60)
print("STEP 3: COMPILING MODEL")
print("="*60)

optimizer = Adam(learning_rate=0.0005)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

print("Model compiled with Adam (lr=0.0005) and categorical_crossentropy.")

# ============================================================
# STEP 4: TRAINING
# ============================================================

EPOCHS = 10

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
]

print("\nTraining started...")
start = time.time()

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=valid_data,
    callbacks=callbacks,
    verbose=1
)

# Training summary
duration = time.time() - start
print(f"\nTraining completed in {duration/60:.1f} min")
print(f"Final Val Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")

# ============================================================
# STEP 5: SAVE MODEL & VISUALIZE RESULTS
# ============================================================

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

model.save('models/aircraft_defect_model.keras')
print("\nModel saved to models/aircraft_defect_model.keras")

# --- Performance plots ---
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(True, alpha=0.3)

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/model_performance.png', dpi=300)
plt.close()
print("Training plots saved to outputs/model_performance.png")

# ============================================================
# STEP 6: FINAL EVALUATION
# ============================================================

train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
acc_gap = train_acc - val_acc

print("\nFinal Training Accuracy:", f"{train_acc*100:.2f}%")
print("Final Validation Accuracy:", f"{val_acc*100:.2f}%")

if acc_gap > 0.15:
    print("\n⚠ Overfitting detected! Consider:")
    print("   - Increasing dropout or augmentation")
    print("   - Simplifying architecture")
    print("   - Adding more training data")
else:
    print("\n✓ Model generalizes well!")

print("\nPipeline complete! Check outputs/ and models/ folders for results.")
