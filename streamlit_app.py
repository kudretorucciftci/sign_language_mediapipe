import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import av
import threading

# MediaPipe Import Fix for some environments
if not hasattr(mp, 'solutions'):
    try:
        from mediapipe.python import solutions as mp_solutions
        mp.solutions = mp_solutions
    except Exception:
        pass

# Sayfa Ayarları
st.set_page_config(page_title="AI Sign Language Translator", layout="wide", page_icon="🤟")

# CSS ile Premium Görünüm
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e1e2f, #121212);
    }
    
    .main-title {
        background: linear-gradient(90deg, #00DBDE 0%, #FC00FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
        padding-top: 20px;
    }
    
    .sub-title {
        color: #888;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }
    
    .prediction-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 30px;
        text-align: center;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        transition: all 0.3s ease;
    }
    
    .prediction-label {
        color: #aaa;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }
    
    .prediction-value {
        color: #00DBDE;
        font-size: 6rem;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(0, 219, 222, 0.4);
    }
    
    .status-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 50px;
        background: rgba(0, 219, 222, 0.1);
        color: #00DBDE;
        font-size: 0.8rem;
        margin-top: 15px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(20, 20, 25, 0.8);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">AI Sign Language</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">MediaPipe & Deep Learning Destekli Gerçek Zamanlı Çeviri</p>', unsafe_allow_html=True)

# Model ve Kaynak Yükleme
@st.cache_resource
def load_resources():
    # Model yükleme
    model = load_model("sign_lang_model.keras")
    harfler = [chr(i) for i in range(65, 91) if i != 74] # A-Z except J
    
    # MediaPipe el tespiti
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=1, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
    return model, harfler, hands, mp_draw

model, harfler, hands, mp_draw = load_resources()

# Thread-safe prediction storage
lock = threading.Lock()
prediction_state = {"char": "-", "confidence": 0.0}

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Görüntü işleme
        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)
        
        current_pred = "-"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Landmarkları çiz
                mp_draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Koordinatları topla
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                
                if len(features) == 63:
                    # Model tahmini
                    input_data = np.array([features])
                    prediction = model.predict(input_data, verbose=0)
                    idx = np.argmax(prediction)
                    conf = np.max(prediction)
                    
                    if conf > 0.5:
                        current_pred = harfler[idx]
                        with lock:
                            prediction_state["char"] = current_pred
                            prediction_state["confidence"] = conf
        else:
            with lock:
                prediction_state["char"] = "El Yok"
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="sign-language",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.markdown('<p class="prediction-label">Tahmin Edilen Harf</p>', unsafe_allow_html=True)
    
    # Placeholder for dynamic update
    prediction_placeholder = st.empty()
    
    # Prediction loop if stream is running
    if webrtc_ctx.state.playing:
        with lock:
            char = prediction_state["char"]
        prediction_placeholder.markdown(f'<p class="prediction-value">{char}</p>', unsafe_allow_html=True)
        st.markdown('<div class="status-badge">Sistem Aktif - Tarama Yapılıyor</div>', unsafe_allow_html=True)
    else:
        prediction_placeholder.markdown('<p class="prediction-value">-</p>', unsafe_allow_html=True)
        st.markdown('<div class="status-badge" style="color: #ff4b2b; background: rgba(255, 75, 43, 0.1);">Kamera Kapalı</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Bilgi Bölümü
st.sidebar.title("Hakkında")
st.sidebar.markdown("""
Bu proje **MediaPipe** ve **TensorFlow** kullanılarak geliştirilmiştir. 
Amerikan İşaret Dili (ASL) alfabesini tanımak için eğitilmiş bir derin öğrenme modeli kullanır.

**Nasıl Kullanılır?**
1. Kameranızı başlatın.
2. Elinizi kameraya net bir şekilde gösterin.
3. Model harfi otomatik olarak tanıyacaktır.
""")

st.sidebar.warning("Not: En iyi sonuç için iyi aydınlatılmış bir ortam tercih edin.")
