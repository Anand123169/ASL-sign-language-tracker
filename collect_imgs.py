import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 🔥 All 26 letters automatically
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

dataset_size = 100

cap = cv2.VideoCapture(0)

for label in labels:
    label_path = os.path.join(DATA_DIR, label)

    if not os.path.exists(label_path):
        os.makedirs(label_path)

    print(f'Collecting data for letter {label}')

    # Waiting screen
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        frame = cv2.flip(frame, 1)

        cv2.putText(frame, f'Ready for {label}? Press Q', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    # Capture images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        frame = cv2.flip(frame, 1)

        cv2.putText(frame, f'{label}: {counter}/{dataset_size}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        cv2.imwrite(os.path.join(label_path, f'{counter}.jpg'), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()