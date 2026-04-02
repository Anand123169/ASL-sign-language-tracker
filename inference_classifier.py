import pickle
import cv2
import mediapipe as mp
import numpy as np
import collections

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Camera
cap = cv2.VideoCapture(0)

# Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# 🔥 Word builder
word = ""
current_letter = ""

# 🔥 Stability buffer
pred_buffer = collections.deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        x_ = []
        y_ = []
        data_aux = []

        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            pred_buffer.append(prediction[0])

            # 🔥 Most common prediction (stability)
            if len(pred_buffer) == 10:
                current_letter = max(set(pred_buffer), key=pred_buffer.count)

            # Bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            cv2.putText(frame, current_letter, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

    # 🔥 Display word
    cv2.putText(frame, f'Word: {word}', (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    # 🔥 Instructions
    cv2.putText(frame, 'SPACE: Add | BACKSPACE: Delete | C: Clear | ESC: Exit',
                (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Sign Language Detector', frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        word += current_letter
    elif key == 8:  # BACKSPACE
        word = word[:-1]
    elif key == ord('c'):
        word = ""

cap.release()
cv2.destroyAllWindows()