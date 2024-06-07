from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Mevcut çalışma dizininden models klasörü altındaki dosyayı yükle
model_path = os.path.join('models', 'workout_model.keras')

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

label_dict = { "barbell biceps curl": 0,
        "bench press": 1,
        "chest fly machine": 2,
        "deadlift": 3,
        "decline bench press": 4,
        "hammer curl": 5,
        "hip thrust": 6,
        "incline bench press": 7,
        "lat pulldown": 8,
        "lateral raises": 9,
        "leg extension": 10,
        "leg raises": 11,
        "plank": 12,
        "pull up": 13,
        "push up": 14,
        "romanian deadlift": 15,
        "russian twist": 16,
        "shoulder press": 17,
        "squat": 18,
        "t bar row": 19,
        "tricep dips": 20,
        "tricep pushdown": 21 }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Hata ayıklama için tahmin edilen sınıfı ve sözlükte olup olmadığını kontrol edin
    print(f"Tahmin edilen sınıf: {predicted_class}")
    if predicted_class in label_dict:
        exercise = label_dict[predicted_class]
    else:
        exercise = "Unknown Exercise"
        print(f"Error: Tahmin edilen sınıf {predicted_class} label_dict sözlüğünde bulunamadı.")

    return jsonify({'exercise': exercise})


    #exercise = label_dict[np.argmax(prediction)]
    #return jsonify({'exercise': exercise})

if __name__ == '__main__':
    app.run(debug=True)
