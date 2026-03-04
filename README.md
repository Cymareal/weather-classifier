# 🌤️ Weather Classifier API

A deep learning image classification model that identifies weather conditions from images. Built with a CNN using MobileNetV2 transfer learning and deployed as a REST API with Flask.

## 🌦️ Weather Classes
- Cloudy
- Rain
- Shine (Sunny)
- Sunrise

## 🚀 Live API
Base URL: `https://your-render-url.onrender.com`

### Predict Endpoint
**POST** `/predict`

Upload an image and get a weather prediction back.

**Example request:**
```bash
curl -X POST -F "file=@your_image.jpg" https://your-render-url.onrender.com/predict
```

**Example response:**
```json
{
  "prediction": "Rain",
  "confidence": 98.36
}
```

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- MobileNetV2 (transfer learning, pretrained on ImageNet)
- Flask
- Deployed on Render

## 🧠 How It Works
1. Images are resized to 224x224 and normalized
2. MobileNetV2 (pretrained) acts as a feature extractor
3. Custom classification layers are trained on top
4. Grad-CAM heatmaps show which parts of the image influenced the prediction
5. Model is served via a Flask REST API

## 📊 Model Performance
- Trained with data augmentation (rotation, zoom, horizontal flip)
- Early stopping to prevent overfitting
- Evaluated with confusion matrix and classification report

## 📁 Project Structure
```
weather-classifier/
├── app.py                        # Flask API
├── weather_model.keras           # Trained model
├── weather_classifier.ipynb      # Training notebook
├── requirements.txt
└── Procfile
```

## ⚙️ Run Locally
```bash
git clone https://github.com/Cymareal/weather-classifier.git
cd weather-classifier
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then send a request:
```bash
curl -X POST -F "file=@your_image.jpg" http://127.0.0.1:5000/predict
```

## 🔗 Related Projects
- [Loan Default Prediction](https://github.com/Cymareal/loan-default-prediction)
- [Sentiment Analysis API](https://github.com/Cymareal/sentiment-analysis-api)
