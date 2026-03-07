from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os

app = Flask(__name__)

model = load_model("weather_model.keras")
CLASS_NAMES = ['Cloudy', 'Rain', 'Shine', 'Sunrise']  # update if your folder names differ
IMG_SIZE = 224

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = image.load_img(io.BytesIO(file.read()), target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return jsonify({
        'prediction': predicted_class,
        'confidence': round(confidence * 100, 2)
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Weather Classifier API is running!'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)