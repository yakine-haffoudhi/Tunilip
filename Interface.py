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
OUTPUT_FOLDER = "LipReading"
IMG_SIZE = (64, 64)

# --- Load Model ---
model = load_model(MODEL_PATH)

# --- Session State Initialization ---
if "frame" not in st.session_state:
    st.session_state.frame = None

# --- Functions ---
def detect_lips(frame, face_mesh):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    for f in results.multi_face_landmarks:
        h, w, _ = frame.shape
        pts = np.array([(int(f.landmark[i].x*w), int(f.landmark[i].y*h)) for i in LIPS_LANDMARKS])
        x,y,ww,hh = cv2.boundingRect(pts)
        if ww>0 and hh>0:
            crop = frame[y:y+hh, x:x+ww]
            return cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    return None

# --- Background & Styling ---
def load_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = load_b64("BG (1).png")
st.markdown(f"""
<style>
  .stApp {{ background-image: url('data:image/png;base64,{bg}'); background-size: cover; }}
  .stButton>button {{ background-color: #FF6347; color:#fff; font-size:20px; border-radius:8px; padding:8px 16px; }}
  .stButton>button:hover {{ background-color:#FF4500; }}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
logo = load_b64("Logo.png")
st.sidebar.markdown(f"<div style='text-align:center;'><img src='data:image/png;base64,{logo}' width=200></div>", unsafe_allow_html=True)
st.sidebar.title("Navigation Bar")
tab = st.sidebar.radio("Go to", ["What is TUNILip?","How to Use","Main"])

# --- What is TUNILip ---
if tab=="What is TUNILip?":
    st.markdown("## 📌 What is **TUNILip**?")
    st.markdown("""
    <div style='font-size:18px;'>
      <strong>TUNILip</strong> is an AI-powered app by <strong>IEEE SIGHT ENIT</strong>.<br>
      It recognizes Tunisian dialect words from lip movements to assist those with hearing impairments.
    </div>
    """, unsafe_allow_html=True)
    st.image("sight.jpg", use_container_width=True)
    st.markdown("""
    <div style='font-size:18px;'>
      Combines Computer Vision and Deep Learning to detect and interpret lip movements in real-time or from uploads.
    </div>
    """, unsafe_allow_html=True)

# --- How to Use ---
elif tab=="How to Use":
    st.markdown("## 🚀 How to Use TUNILip")
    st.markdown("1. Click 'Main' tab and take a snapshot or upload an image/video.\n2. Click 'Generate Lip Matrix' to crop lips.\n3. Click 'Get Result' to see the predicted word.")

# --- Main ---
else:
    st.title("TUNILip")
    # camera capture
    img_in = st.camera_input("📷 Capture your lips")
    if img_in:
        data = img_in.read()
        arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        st.session_state.frame = frame
        st.image(frame, caption="Captured frame", width=320)

    if st.button("🧩 Generate Lip Matrix"):
        if st.session_state.frame is not None:
            with st.spinner("Processing..."):
                mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
                lip = detect_lips(st.session_state.frame, mesh)
            if lip is not None:
                st.image(lip, caption="Lip crop", width=320)
                st.session_state.lip = lip
            else:
                st.warning("⚠️ No lips detected.")
        else:
            st.warning("⚠️ Please capture an image first.")

    if st.button("📊 Get Result"):
        if hasattr(st.session_state, 'lip'):
            gray = cv2.cvtColor(st.session_state.lip, cv2.COLOR_BGR2GRAY)
            in_img = gray.reshape((1,)+IMG_SIZE+(1,))/255.0
            pred = model.predict(in_img)
            word = CLASSES[np.argmax(pred)]
            st.success(f"🗣️ Detected Word: **{word}**")
        else:
            st.warning("⚠️ Generate a lip matrix first.")

    st.markdown("<footer style='text-align:center;color:#999;'>Created by IEEE SIGHT ENIT</footer>", unsafe_allow_html=True)
