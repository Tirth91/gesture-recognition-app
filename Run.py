import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle


model = tf.keras.models.load_model("landmark_model.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


CONFIDENCE_THRESHOLD = 0.7  #

def predict_gesture():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Extract landmarks
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks[:2]:  # Max 2 hands
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

      
        while len(landmarks) < 188:
            landmarks.append(0.0)  
        landmarks = landmarks[:188]  

        
        if len(landmarks) == 188 and sum(landmarks) != 0.0:  
            landmarks = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks)
            max_prob = np.max(prediction)  
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

            
            if max_prob < CONFIDENCE_THRESHOLD:
                predicted_label = "Sign Not Recognized"
        else:
            predicted_label = "None"  

        
        cv2.putText(frame, f"Predicted: {predicted_label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("Gesture Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

predict_gesture()
