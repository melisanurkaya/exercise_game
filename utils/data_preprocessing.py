import os

DATASET_DIR = 'workout_data'

labels = os.listdir(DATASET_DIR)
print(labels)

import numpy as np
import cv2
from sklearn.model_selection import train_test_split




DATASET_DIR = 'workout_data'


def load_data(img_size=(64, 64)):
    images = []
    labels = []

    # Etiketleri veri kümesinden öğrenelim
    dataset_labels = os.listdir(DATASET_DIR)

    # Etiketleri label_dict'e ekleyelim
    label_dict = {
        "barbell biceps curl": 0,
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
        "tricep pushdown": 21
    }

    # Verisetindeki tüm etiketleri yazdırmak için bir kez çalıştırabilirsiniz
    for label in dataset_labels:
        print(label)  # Bu satırı tüm etiketleri görmek için ekliyoruz

    for label in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, label)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                images.append(img)
                try:
                    labels.append(label_dict[label])
                except KeyError:
                    print(f"Etiket {label} label_dict içinde bulunamadı.")
                    continue

    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)



    return train_test_split(images, labels, test_size=0.2, random_state=42)


x_train, x_test, y_train, y_test = load_data()
