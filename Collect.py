import cv2
import mediapipe as mp
import numpy as np
import csv
import os


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


DATA_DIR = r"D:\Gesture"
CSV_FILE = os.path.join(DATA_DIR, "gesture_data.csv")


if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["label"] + [f"x{i}" for i in range(63*2)] + ["pad" + str(i) for i in range(62)]  # 188 features
        writer.writerow(header)


def capture_landmarks(label, num_samples=500):
    cap = cv2.VideoCapture(0)
    collected = 0

    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)

        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Extract landmarks
            landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks[:2]:  # Ensure max 2 hands
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

            # Ensure exactly 188 features (Padding or trimming)
            while len(landmarks) < 188:
                landmarks.append(0.0)  # Padding
            landmarks = landmarks[:188]  # Trimming if needed

            writer.writerow([label] + landmarks)
            collected += 1
            print(f"Collected {collected}/{num_samples} for {label}")

            # Display frame
            cv2.putText(frame, f"Collecting: {label} ({collected}/{num_samples})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow("Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Run data collection
label = input("Enter the letter/sign you want to collect: ")
capture_landmarks(label, num_samples=500)
