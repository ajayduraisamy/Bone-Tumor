from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask_cors import CORS
import io

# Load trained model
model = load_model("trained_model_CNN.h5")

# Class labels
class_labels = ["Malignant", "Normal"]

# Image dimensions
img_width, img_height = 224, 224

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Burn Classifier API is running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # ✅ Load image from file-like object instead of path
        img = image.load_img(io.BytesIO(file.read()), target_size=(img_width, img_height))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype("float32") / 255.0

        preds = model.predict(x)
        class_index = np.argmax(preds)
        confidence = float(np.max(preds))

        return jsonify({
            "predicted_class": class_labels[class_index],
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
