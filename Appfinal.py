import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle

# Load Model and Encoder
model = tf.keras.models.load_model("landmark_model.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

CONFIDENCE_THRESHOLD = 0.7


def process_frame(frame):
    """Process the frame and predict gesture."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    # Pad or trim landmarks to 188 elements
    while len(landmarks) < 188:
        landmarks.append(0.0)
    landmarks = landmarks[:188]

    predicted_label = "None"
    if len(landmarks) == 188 and sum(landmarks) != 0.0:
        landmarks = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(landmarks)
        max_prob = np.max(prediction)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

        if max_prob < CONFIDENCE_THRESHOLD:
            predicted_label = "Sign Not Recognized"

    return predicted_label


class VideoProcessor(VideoProcessorBase):
    """Custom video processor to handle real-time frame prediction."""
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        prediction = process_frame(img)

        # Display Prediction on Frame
        cv2.putText(
            img,
            f"Predicted: {prediction}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Streamlit UI
st.set_page_config(page_title="Real-Time Gesture Recognition")
st.title("🤝 Real-Time Gesture Recognition with WebRTC 🎥")

webrtc_ctx = webrtc_streamer(
    key="gesture-recognition",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 10}},
        "audio": False,
    },
)

# Instructions
st.markdown("""
### ✨ How to Use:
- Show gestures in front of the camera.
- Predictions will appear on the video feed.
""")
