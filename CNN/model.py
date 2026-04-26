import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

# ======================================================
# 🔹 DATASET PATH (CHANGE ONLY THIS)
# ======================================================
data_dir = r"C:\Users\venka\Downloads\archive (2)\plantvillage"

print("Dataset Exists:", os.path.exists(data_dir))

# ======================================================
# 🔹 PARAMETERS (OPTIMIZED)
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

    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
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

# Save class names (VERY IMPORTANT for GUI)
class_names = list(train_data.class_indices.keys())
print("Classes:", class_names)

# ======================================================
# 🔹 CNN MODEL
# ======================================================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================================================
# 🔹 TRAIN MODEL
# ======================================================
print("\nTraining CNN Model...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ======================================================
# 🔹 SAVE MODEL (.h5 for GUI)
# ======================================================
model.save("cnn_model.h5")
print("Model saved as cnn_model.h5")

# ======================================================
# 🔹 PLOT RESULTS
# ======================================================
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()