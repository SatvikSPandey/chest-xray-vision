import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\babaw\OneDrive\Desktop\work\Project 11 — Computer Vision\archive\chest_xray"
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
VAL_DIR    = os.path.join(BASE_DIR, "val")
WEIGHTS_DIR = r"C:\Users\babaw\OneDrive\Desktop\work\Project 11 — Computer Vision\weights"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "efficientnet_xray.h5")

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224        # EfficientNetB0 expects 224x224 images
BATCH_SIZE  = 32
EPOCHS      = 15
NUM_CLASSES = 2

# ── Data Augmentation ──────────────────────────────────────────────────────────
# Training data: apply augmentation to improve generalisation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,          # normalise pixel values to [0, 1]
    horizontal_flip=True,          # randomly flip images left-right
    rotation_range=10,             # randomly rotate up to 10 degrees
    zoom_range=0.1,                # randomly zoom in up to 10%
    width_shift_range=0.1,         # randomly shift horizontally
    height_shift_range=0.1         # randomly shift vertically
)

# Validation data: only rescale, no augmentation
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Class indices:", train_generator.class_indices)

# ── Model ──────────────────────────────────────────────────────────────────────
# Load EfficientNetB0 without its top classification layer
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model — we don't retrain ImageNet weights yet
base_model.trainable = False

# Add our own classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)       # flatten feature maps to a vector
x = Dropout(0.3)(x)                   # dropout to prevent overfitting
x = Dense(128, activation='relu')(x)  # intermediate dense layer
x = Dropout(0.2)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)  # final 2-class output

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ──────────────────────────────────────────────────────────────────
callbacks = [
    # Save weights only when validation accuracy improves
    ModelCheckpoint(
        WEIGHTS_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Stop training early if val_accuracy stops improving for 5 epochs
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate if val_accuracy plateaus
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

# ── Phase 1: Train classification head only ────────────────────────────────────
print("\n=== Phase 1: Training classification head ===")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ── Phase 2: Fine-tune top layers of base model ────────────────────────────────
print("\n=== Phase 2: Fine-tuning top layers of EfficientNet ===")

# Unfreeze the last 30 layers of the base model
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with a much lower learning rate to avoid destroying learned weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print(f"\nTraining complete. Best weights saved to: {WEIGHTS_PATH}")