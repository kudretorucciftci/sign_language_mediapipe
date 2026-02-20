import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import av
import threading
import mediapipe as mp
import time

# MediaPipe bileşenlerini güvenli bir şekilde al
try:
    # En güvenli yol: Doğrudan modül yollarından import
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw
    print("LOG: MediaPipe modules loaded via direct python path.")
except ImportError:
    try:
        # Alternatif yol: Standart solutions üzerinden
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        print("LOG: MediaPipe modules loaded via standard solutions path.")
    except (AttributeError, ImportError) as e:
        st.error(f"MediaPipe Modül Hatası: {e}")
        st.info("Sistem kütüphaneleri hazırlanıyor olabilir, lütfen sayfayı yenileyin veya 'Reboot app' yapın.")
        st.stop()

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
    
    # MediaPipe el tespiti (global mp_hands kullanılıyor)
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=1, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return model, harfler, hands

model, harfler, hands = load_resources()

# Thread-safe prediction storage
lock = threading.Lock()
prediction_state = {"char": "-", "confidence": 0.0}

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_pred = "-"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Görüntü işleme
        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # MediaPipe işleme (her frame'de yapılabilir, hafiftir)
        results = hands.process(rgb_img)
        
        current_pred = "Taraniyor..."
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Landmarkları çiz
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Sadece her 5 frame'de bir ağır model tahmini yap
                if self.frame_count % 5 == 0:
                    features = []
                    for lm in hand_landmarks.landmark:
                        features.extend([lm.x, lm.y, lm.z])
                    
                    if len(features) == 63:
                        input_data = np.array([features])
                        prediction = model.predict(input_data, verbose=0)
                        idx = np.argmax(prediction)
                        conf = np.max(prediction)
                        
                        if conf > 0.4: # Teşhis eşiğini biraz düşürdük
                            self.last_pred = harfler[idx]
                            with lock:
                                prediction_state["char"] = self.last_pred
                                prediction_state["confidence"] = conf
                
                # Video üzerine yazdır (Anlık geri bildirim)
                cv2.putText(img, f"Harf: {self.last_pred}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 219, 222), 3)
        else:
            with lock:
                prediction_state["char"] = "El Yok"
                self.last_pred = "-"
        
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
    
    # Placeholderlar
    prediction_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Canlı Güncelleme Döngüsü (Güvenli yapı)
    if webrtc_ctx.state.playing:
        status_placeholder.markdown('<div class="status-badge">Sistem Aktif - Tarama Yapılıyor</div>', unsafe_allow_html=True)
        try:
            while webrtc_ctx.state.playing:
                with lock:
                    char = prediction_state["char"]
                    conf = prediction_state["confidence"]
                
                prediction_placeholder.markdown(f'<p class="prediction-value">{char}</p>', unsafe_allow_html=True)
                time.sleep(0.5) # UI yükünü azaltmak için süreyi artırdık
        except Exception:
            pass
    else:
        prediction_placeholder.markdown('<p class="prediction-value">-</p>', unsafe_allow_html=True)
        status_placeholder.markdown('<div class="status-badge" style="color: #ff4b2b; background: rgba(255, 75, 43, 0.1);">Kamera Kapalı</div>', unsafe_allow_html=True)
    
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
