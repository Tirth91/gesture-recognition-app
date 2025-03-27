import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# ------------------------------
# 🎯 Load Model and Encoder
# ------------------------------
model = tf.keras.models.load_model("landmark_model.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ------------------------------
# 🖐️ Initialize MediaPipe Hands
# ------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for gesture prediction

# ------------------------------
# 🎥 Video Frame Processing
# ------------------------------
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")  # Convert frame to numpy array
    img = cv2.resize(img, (480, 360))  # Resize to 480x360 for better performance
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe

    results = hands.process(rgb_frame)

    # Extract landmarks
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:2]:  # Max 2 hands
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    # Pad or trim landmarks to 188 elements
    while len(landmarks) < 188:
        landmarks.append(0.0)
    landmarks = landmarks[:188]

    # Predict Gesture Only if Landmarks Are Detected
    predicted_label = "None"
    if len(landmarks) == 188 and sum(landmarks) != 0.0:
        landmarks = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(landmarks)
        max_prob = np.max(prediction)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

        if max_prob < CONFIDENCE_THRESHOLD:
            predicted_label = "Sign Not Recognized"

    # Display prediction on frame
    cv2.putText(img, f"Predicted: {predicted_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------------------
# 🎨 Streamlit UI
# ------------------------------
st.set_page_config(page_title="Real-Time Gesture Recognition")
st.title("Real-Time Gesture Recognition with WebRTC 🎥🤝")

# 🔁 Start WebRTC Video
webrtc_ctx = webrtc_streamer(
    key="gesture-recognition",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 10}},
        "audio": False,
    },
)

# 📝 Instructions
st.markdown("""
### 🎉 How to Use:
- Show gestures in front of the camera.
- Predictions will appear on the video feed.
""")
