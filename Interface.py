import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from datetime import datetime
import base64
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# --- Global Parameters ---
LIPS_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
    37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82,
    81, 42, 183, 78
]
CLASSES = [
    "Aaslema", "Aatchan", "Ghatini", "Hamdoulah", "Inchallah", "Ji3an", "Mahsour",
    "Mawjou3", "Met9alla9", "Nadhafli", "Skhont", "Aychek", "Yezzini"
]
MODEL_PATH = "model.h5"
OUTPUT_FOLDER = "LipReading"
IMG_SIZE = (64, 64)

# --- Load Model ---
model = load_model(MODEL_PATH)

# --- Session State ---
if "lip_frame" not in st.session_state:
    st.session_state.lip_frame = None

# --- Processor for WebRTC ---
class LipVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.lip_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # detect lips
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            pts = np.array([(int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for lm in results.multi_face_landmarks[0].landmark if lm.visibility > 0.5])
            x,y,w,h = cv2.boundingRect(pts)
            self.lip_frame = img[y:y+h, x:x+w]
            if w>0 and h>0:
                crop = cv2.resize(self.lip_frame, IMG_SIZE)
                self.lip_frame = crop
        st.session_state.lip_frame = self.lip_frame
        return frame

# --- Sidebar ---
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["What is TUNILip?", "How to Use", "Main"])

# --- What is TUNILip ---
if tab == "What is TUNILip?":
    st.markdown("## What is TUNILip?")
    st.markdown("**TUNILip** is an AI-powered app by IEEE SIGHT ENIT that recognizes Tunisian dialect words from lip movements.")
    st.image("sight.jpg", use_container_width=True)

# --- How to Use ---
elif tab == "How to Use":
    st.markdown("## How to Use TUNILip")
    st.markdown("1. Go to Main. 2. Click 'Start Webcam'. 3. Click 'Generate Lip Matrix'. 4. Click 'Get Result'.")

# --- Main Tab ---
else:
    st.title("TUNILip")
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="lip_webcam",
        video_processor_factory=LipVideoProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

    if st.button("🧩 Generate Lip Matrix"):
        if st.session_state.lip_frame is not None:
            st.image(st.session_state.lip_frame, caption="Lip Crop", width=320)
            matrix_path = os.path.join(OUTPUT_FOLDER, "live_lip.jpg")
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            cv2.imwrite(matrix_path, st.session_state.lip_frame)
            st.session_state.matrix_path = matrix_path
        else:
            st.warning("⚠️ No lip frame captured yet.")

    if st.button("📊 Get Result"):
        if "matrix_path" in st.session_state:
            img = cv2.imread(st.session_state.matrix_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inp = gray.reshape((1,)+IMG_SIZE+(1,))/255.0
            pred = model.predict(inp)
            word = CLASSES[np.argmax(pred)]
            st.success(f"Detected Word: **{word}**")
        else:
            st.warning("⚠️ Generate the lip matrix first.")

    st.markdown("<footer style='text-align:center;color:#999;'>Created by IEEE SIGHT ENIT</footer>", unsafe_allow_html=True)
