import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('trained_model_CNN.h5')

# Class labels (update these with your actual class names)
burn_classes = ['malignant', 'Normal'] 
# Image dimensions
img_width, img_height = 224, 224

# Function to process image and predict
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32') / 255
    prediction = model.predict(x)
    predicted_class = np.argmax(prediction)
    return burn_classes[predicted_class]

# Function to open file dialog and display prediction
def open_and_classify():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        # Classify and display result
        result = classify_image(file_path)
        result_label.config(text=f"Prediction: {result}")

# Create main window
root = tk.Tk()
root.title("CNN Burn Classifier")
root.geometry("400x400")

# UI Elements
panel = Label(root)
panel.pack(pady=10)

btn = Button(root, text="Choose Image and Predict", command=open_and_classify)
btn.pack(pady=10)

result_label = Label(root, text="Prediction: ", font=("Arial", 14))
result_label.pack(pady=10)

# Start GUI loop
root.mainloop()