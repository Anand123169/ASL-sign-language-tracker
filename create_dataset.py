import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

DATA_DIR = './data'

data = []
labels = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(dir_path)[:1000]:
        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # 🔥 only 1 hand

            x_ = []
            y_ = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"✅ Dataset created with {len(data)} samples")