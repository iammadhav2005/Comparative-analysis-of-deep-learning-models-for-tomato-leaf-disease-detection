import tensorflow as tf
import numpy as np
import cv2
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# ======================================================
# LOAD MODEL
# ======================================================
model = tf.keras.models.load_model("cnn_model.h5")

# ======================================================
# PARAMETERS
# ======================================================
IMG_SIZE = 160

# MUST MATCH YOUR DATASET FOLDERS
class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Tomato___Leaf_Mold"
]

# ======================================================
# CREATE WINDOW
# ======================================================
root = Tk()
root.title("Tomato Leaf Disease Detection")
root.geometry("500x600")

# ======================================================
# LABELS
# ======================================================
label = Label(root, text="Upload Tomato Leaf Image", font=("Arial", 16))
label.pack(pady=10)

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=20)

# ======================================================
# FUNCTION TO PREDICT
# ======================================================
def upload_image():
    file_path = filedialog.askopenfilename()

    if not file_path:
        return

    # Show image in GUI
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    # Preprocess image
    img_cv = cv2.imread(file_path)
    img_cv = cv2.resize(img_cv, (IMG_SIZE, IMG_SIZE))
    img_cv = img_cv / 255.0
    img_cv = np.expand_dims(img_cv, axis=0)

    # Prediction
    pred = model.predict(img_cv)
    class_index = np.argmax(pred)
    confidence = np.max(pred)

    # Show result
    result = f"Prediction: {class_names[class_index]}\nConfidence: {round(confidence*100, 2)}%"
    result_label.config(text=result)

# ======================================================
# BUTTON
# ======================================================
btn = Button(root, text="Upload Image", command=upload_image, font=("Arial", 14))
btn.pack(pady=20)

# ======================================================
# RUN GUI
# ======================================================
root.mainloop()