import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

# ======================================================
# 🔹 DATASET PATH
# ======================================================
data_dir = r"C:\Users\venka\Downloads\archive (2)\plantvillage"

print("Dataset Exists:", os.path.exists(data_dir))

# ======================================================
# 🔹 PARAMETERS
# ======================================================
IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 5

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
# 🔹 MODEL
# ======================================================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation='relu')(x)
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
print("\nTraining MobileNetV2...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ======================================================
# 🔹 SAVE MODEL (.h5)
# ======================================================
model.save("mobilenet_model.h5")
print("✅ Model saved as mobilenet_model.h5")

# ======================================================
# 🔹 PLOT
# ======================================================
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('MobileNetV2 Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MobileNetV2 Loss')
plt.legend(['Train', 'Validation'])
plt.show()