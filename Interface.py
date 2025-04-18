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
GRID_SIZE = (4, 10)

# --- Load Model ---
model = load_model(MODEL_PATH)

# --- Session State Initialization ---
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "matrix_path" not in st.session_state:
    st.session_state.matrix_path = None
if "stop" not in st.session_state:
    st.session_state.stop = False

# --- Functions ---
def detect_lips(frame, face_mesh):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        return None
    for face_landmarks in results.multi_face_landmarks:
        h, w, _ = frame.shape
        lip_points = np.array([
            (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
            for idx in LIPS_LANDMARKS
        ])
        x, y, w_, h_ = cv2.boundingRect(lip_points)
        if w_ > 0 and h_ > 0:
            lips = frame[y:y + h_, x:x + w_]
            lips = cv2.resize(lips, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
            return lips if lips.size > 0 else None
    return None

def process_video(video_path, output_folder, grid_size=GRID_SIZE, img_size=IMG_SIZE):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    video_name = os.path.basename(video_path).split('.')[0]
    lip_images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        lip_frame = detect_lips(frame, face_mesh)
        if lip_frame is not None:
            resized = cv2.resize(lip_frame, img_size, interpolation=cv2.INTER_CUBIC)
            lip_images.append(resized)
        if len(lip_images) >= grid_size[0] * grid_size[1]:
            break
    cap.release()
    if lip_images:
        while len(lip_images) < grid_size[0] * grid_size[1]:
            lip_images.append(lip_images[0])
        rows = [
            np.hstack(lip_images[i * grid_size[1]:(i + 1) * grid_size[1]])
            for i in range(grid_size[0])
        ]
        grid_image = np.vstack(rows)
        grid_image = cv2.resize(grid_image, (320, 320))  # Smaller matrix
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{video_name}_lips.jpg")
        cv2.imwrite(output_path, grid_image)
        return output_path
    return None

def load_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# --- Background Setup ---
background_image_path = "BG (1).png"
background_base64 = load_image_as_base64(background_image_path)

st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{background_base64}");
            background-size: cover;
            background-position: center;
        }}
        .stButton>button {{
            background-color: #FF6347;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 20px;
        }}
        .stButton>button:hover {{
            background-color: #FF4500;
        }}
    </style>
    """, unsafe_allow_html=True
)

# --- Sidebar ---
logo_path = "Logo.png"
logo_base64 = load_image_as_base64(logo_path)
st.sidebar.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{logo_base64}" width="250" />
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("Navigation Bar")
tab = st.sidebar.radio("Go to", ["How to Use", "Main"])

# --- Main Tab ---
if tab == "Main":
    st.title("TUNILip")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎥 Start Video"):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
            st.session_state.video_path = tmp_file.name

            cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(tmp_file.name, fourcc, 20.0, (640, 480))

            stframe = st.empty()
            st.info("🎥 Recording for 2 seconds...")

            frame_count = 0
            max_frames = 2 * 20  # 2 seconds at 20 fps
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                stframe.image(frame, channels="BGR", use_container_width=False, width=320)
                frame_count += 1

            cap.release()
            out.release()
            st.success("✅ Video recorded for 2 seconds successfully.")

    video_file = st.file_uploader("📤 Upload a video", type=["mp4", "avi"])
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(video_file.read())
            st.session_state.video_path = tmp_file.name
        st.success("Video uploaded successfully!")

    if st.button("🧩 Generate Lip Matrix"):
        if st.session_state.video_path:
            with st.spinner("Generating Lip Matrix..."):
                matrix_path = process_video(st.session_state.video_path, OUTPUT_FOLDER)
            if matrix_path:
                st.session_state.matrix_path = matrix_path
                st.image(matrix_path, caption="🧠 Lip Matrix", use_container_width=False, width=320)
            else:
                st.warning("⚠️ No lips detected.")
        else:
            st.warning("⚠️ Please record or upload a video first.")

    if st.button("📊 Get Result"):
        if st.session_state.matrix_path:
            img = cv2.imread(st.session_state.matrix_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMG_SIZE)
            img = np.expand_dims(img, axis=-1) / 255.0
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)
            predicted_word = CLASSES[predicted_class]
            st.success(f"🗣️ Detected Word: **{predicted_word}**")
        else:
            st.warning("⚠️ Generate the lip matrix first.")

    st.markdown("""
    <footer style="text-align: center; font-size: 20px; color: #999;">
        Created by Yakine Haffoudhi and Manar Aljene
    </footer>
    """, unsafe_allow_html=True)

# --- How to Use Tab ---
else:
    st.markdown("## 1️⃣ What is **TUNILip**?")
    st.markdown("""
    **TUNILip** (Tunisian Dialect Lip Reading) is an AI-powered application created by **Yakine Haffoudhi** and **Manar Aljene**.  
    It helps people with hearing impairments by recognizing Tunisian dialect words from lip movements.
    """)

    st.markdown("## 2️⃣ How It Works 🎥")
    st.video("Tunilip.mp4")

    st.markdown("## 3️⃣ Tips 📝")
    st.markdown("""
    - Ensure proper lighting.
    - Look directly at the camera.
    - Avoid background distractions.
    - Speak slowly and clearly.
    """)

    st.markdown("## 4️⃣ Let’s Try! 🚀")
    if st.button("Go to Main Page"):
        st.experimental_set_query_params(page="main")
        st.rerun()





