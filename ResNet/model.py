import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

# ======================================================
# 🔹 DATASET PATH
# ======================================================
data_dir = r"C:\Users\venka\Downloads\archive (2)\plantvillage"

print("Dataset Exists:", os.path.exists(data_dir))

# ======================================================
# 🔹 OPTIMIZED PARAMETERS (FASTER 🔥)
# ======================================================
IMG_SIZE = 160   # reduced from 224
BATCH_SIZE = 16  # reduced from 32
EPOCHS = 5       # enough for comparison

# ======================================================
# 🔹 DATA AUGMENTATION
# ======================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# ======================================================
# 🔹 LOAD DATA
# ======================================================
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = val_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ======================================================
# 🔹 RESNET50 MODEL
# ======================================================
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze ALL layers (important for speed)
base_model.trainable = False

# Custom layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation='relu')(x)  # smaller layer
x = layers.Dropout(0.5)(x)
output = layers.Dense(len(train_data.class_indices), activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# ======================================================
# 🔹 COMPILE
# ======================================================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ======================================================
# 🔹 TRAIN
# ======================================================
print("\nTraining Fast ResNet50...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ======================================================
# 🔹 PLOT
# ======================================================
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('ResNet50 Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ResNet50 Loss')
plt.legend(['Train', 'Validation'])
plt.show()