# arayuz_test.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk

# Harf listesi (A-Z, J harfini atladýk)
harfler = [chr(i) for i in range(65, 91) if i != 74]

# Modeli yükle
model = load_model("mediapipe_model_v4.keras")

# Mediapipe eller modülü
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Tahmin fonksiyonu
def tahmin_et(landmarks):
    # 63 özellik: 21 nokta × (x, y, z)
    vektor = []
    for lm in landmarks.landmark:
        vektor.extend([lm.x, lm.y, lm.z])
    vektor = np.array(vektor).reshape(1, -1)  # (1, 63)
    tahmin = model.predict(vektor, verbose=0)
    indeks = np.argmax(tahmin)
    return harfler[indeks]

# Arayüz
window = tk.Tk()
window.title("El Hareketi Tanima")
window.geometry("800x600")

# Etiket
etiket = tk.Label(window, text="Tahmin edilen harf", font=("Arial", 24))
etiket.pack()

# Görüntü alaný
panel = tk.Label(window)
panel.pack()

# Kamera akýþý
cap = cv2.VideoCapture(0)

def kamera_akisi():
    ret, frame = cap.read()
    if not ret:
        window.after(10, kamera_akisi)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    tahmin = "Yok"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            tahmin = tahmin_et(handLms)

    # Tahmini ekranda göster
    etiket.config(text=f"Tahmin: {tahmin}")

    # Görüntüyü arayüze aktar
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.configure(image=imgtk)

    window.after(10, kamera_akisi)

kamera_akisi()
window.mainloop()
cap.release()
