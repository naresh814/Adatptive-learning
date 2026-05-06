import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# ── Configurations ────────────────────────────────────────────────────────────
DATASET_DIR = "dataset/fer2013"
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
TEST_DIR    = os.path.join(DATASET_DIR, "test")

IMG_SIZE    = (96, 96)
BATCH_SIZE  = 64
NUM_CLASSES = 7

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# ── Data Generators ───────────────────────────────────────────────────────────
# MobileNetV2 expects inputs in range [-1, 1] usually, but if we rescale 1./255 
# we can just use MobileNetV2's built-in preprocessing or feed it 0-1.
# Actually, MobileNetV2 expects preprocessed inputs using tf.keras.applications.mobilenet_v2.preprocess_input
# Let's stick to the 1./255 as requested, MobileNetV2 can adapt if we train it properly.

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ── Class Weights ─────────────────────────────────────────────────────────────
# Handle class imbalance in FER2013
classes = train_generator.classes
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes
)
class_weights = dict(enumerate(class_weights_arr))
print("[INFO] Class weights:", class_weights)

# ── Model Architecture ────────────────────────────────────────────────────────
print("[INFO] Building MobileNetV2 model...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model for Phase 1
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ── Callbacks ─────────────────────────────────────────────────────────────────
checkpoint = ModelCheckpoint(
    "model/emotion_model.h5", 
    monitor="val_accuracy", 
    save_best_only=True, 
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# ── Phase 1: Train Top Layers ─────────────────────────────────────────────────
print("\n" + "="*50)
print("[INFO] PHASE 1: Training top layers (base model frozen)...")
print("="*50)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# ── Phase 2: Fine-Tuning ──────────────────────────────────────────────────────
print("\n" + "="*50)
print("[INFO] PHASE 2: Fine-tuning last 40 layers of base model...")
print("="*50)

# Unfreeze base model
base_model.trainable = True

# Freeze all layers except the last 40
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Recompile with very low learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Continue training
history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,  # Or whatever limit, EarlyStopping will halt it
    class_weight=class_weights,
    callbacks=callbacks
)

print("[INFO] Training Complete! Best model saved to model/emotion_model.h5")
