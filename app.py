from flask import Flask, request, jsonify
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os

def build_model():
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

model = build_model()
model.load_weights("weather_model.weights.h5")

app = Flask(__name__)
CLASS_NAMES = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
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