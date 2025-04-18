import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from datetime import datetime
import base64

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
IMG_SIZE = (64, 64)
GRID_SIZE = (4, 10)

# --- Load Model ---
model = load_model(MODEL_PATH)

# --- Session State Initialization ---
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "matrix_image" not in st.session_state:
    st.session_state.matrix_image = None

# --- Functions ---
def detect_lips(frame, face_mesh):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    for lm in results.multi_face_landmarks:
        h, w, _ = frame.shape
        pts = np.array([(int(l.landmark[i].x * w), int(l.landmark[i].y * h)) 
                        for i, l in [(idx, lm) for idx in LIPS_LANDMARKS]])
        x, y, w_, h_ = cv2.boundingRect(pts)
        if w_ > 0 and h_ > 0:
            crop = frame[y:y+h_, x:x+w_]
            return cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    return None

# Process video from path, return grid numpy image
def process_video(path, grid_size=GRID_SIZE, img_size=IMG_SIZE):
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    lip_images = []
    while len(lip_images) < grid_size[0] * grid_size[1]:
        ret, frame = cap.read()
        if not ret:
            break
        lip = detect_lips(frame, mp_face_mesh)
        if lip is not None:
            lip_images.append(lip)
    cap.release()
    if not lip_images:
        return None
    # pad
    while len(lip_images) < grid_size[0] * grid_size[1]:
        lip_images.append(lip_images[0])
    rows = [np.hstack(lip_images[i*grid_size[1]:(i+1)*grid_size[1]]) for i in range(grid_size[0])]
    grid = np.vstack(rows)
    grid = cv2.resize(grid, (320,320))
    return grid

# --- Streamlit Interface ---
st.title("TUNILip")

tab = st.sidebar.radio("Go to", ["How to Use", "Main"])

if tab == "Main":
    st.title("TUNILip")
    col1, col2 = st.columns(2)
    with col1:
        # Use camera_input for video capture
        vid = st.camera_input("🎥 Record Video (max 10s)")
        if vid:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tmp.write(vid.read())
            tmp.flush()
            st.session_state.video_path = tmp.name
            st.success("✅ Video recorded successfully.")

    # fallback upload
    upload = st.file_uploader("📤 Or upload a video", type=["mp4","avi"])
    if upload:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload.name)[1])
        tmp.write(upload.read()); tmp.flush()
        st.session_state.video_path = tmp.name
        st.success("✅ Video uploaded successfully.")

    if st.button("🧩 Generate Lip Matrix"):
        path = st.session_state.video_path
        if path:
            with st.spinner("Processing video..."):
                matrix = process_video(path)
            if matrix is not None:
                st.session_state.matrix_image = matrix
                st.image(matrix, caption="🧠 Lip Matrix", width=320)
            else:
                st.warning("⚠️ No lips detected.")
        else:
            st.warning("⚠️ Record or upload a video first.")

    if st.button("📊 Get Result"):
        if st.session_state.matrix_image is not None:
            gray = cv2.cvtColor(st.session_state.matrix_image, cv2.COLOR_BGR2GRAY)
            inp = gray.reshape((1,)+IMG_SIZE+(1,))/255.0
            pred = model.predict(inp)
            word = CLASSES[np.argmax(pred)]
            st.success(f"🗣️ Detected Word: **{word}**")
        else:
            st.warning("⚠️ Generate the lip matrix first.")

    st.markdown("""
    <footer style="text-align: center; font-size: 20px; color: #999;">
        Created by IEEE SIGHT ENIT
    </footer>
    """, unsafe_allow_html=True)

else:
    st.markdown("## 1️⃣ What is **TUNILip**?")
    st.markdown("""
        <div style="font-size: 20px;">
            <strong>TUNILip</strong> is an innovative AI-powered application created by <strong>IEEE SIGHT ENIT</strong>.<br>
            It aims to recognize Tunisian dialect words from lip movements, offering accessibility tools for people with hearing impairments.
        </div>
    """, unsafe_allow_html=True)
    st.image("sight.jpg", use_container_width=True)
    st.markdown("""
        <div style="font-size: 20px;">
            This project combines Computer Vision and Deep Learning techniques to detect and interpret lip movements in real-time videos or uploaded files.
        </div>
    """, unsafe_allow_html=True)
    st.markdown("## 2️⃣ How It Works 🎥")
    st.video("Tunilip.mp4")
    st.markdown("## 3️⃣ Tips 📝")
    st.markdown("""
    <div style="font-size: 20px;">
    - Ensure proper lighting.<br>
    - Look directly at the camera.<br>
    - Avoid background distractions.<br>
    - Speak slowly and clearly.<br>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("## 4️⃣ Let’s Try! 🚀")
    st.button("Go to Main Page")
