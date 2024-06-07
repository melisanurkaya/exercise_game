import cv2
import numpy as np
from keras.models import load_model

# Modeli yükle
model = load_model('workout_model.keras')

# Etiketlerin tanımlanması
label_dict = {0: "Elbow Plank", 1: "High Knees", 2: "Squad", 3: "Punches", 4: "Leg Curls", 5: "Superman",
              6: "Chest Squeezes"}

# Video akışı yakalama
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Giriş resmini modelin beklentilerine uygun hale getirme ve normalize etme
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img, axis=0) / 255.0

    # Model üzerinden tahmin yapma
    prediction = model.predict(img)
    exercise = label_dict[np.argmax(prediction)]

    # Tahmini ekrana yazdırma
    cv2.putText(frame, exercise, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görüntüyü gösterme
    cv2.imshow('Exercise Detection', frame)

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma ve pencereyi kapatma
cap.release()
cv2.destroyAllWindows()
