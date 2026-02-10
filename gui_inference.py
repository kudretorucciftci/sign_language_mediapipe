# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys

# Suppress TensorFlow logs (Errors only)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("LOG: Loading Libraries (Keras 3 & MediaPipe)...")

try:
    from tensorflow.keras.models import load_model
    import tkinter as tk
    from PIL import Image, ImageTk
    
    # MediaPipe Import Fix
    try:
        import mediapipe as mp
        if not hasattr(mp, 'solutions'):
            from mediapipe.python import solutions as mp_solutions
            mp.solutions = mp_solutions
        print("✅ LOG: MediaPipe connected successfully.")
    except Exception as e:
        print(f"❌ LOG: MediaPipe connection issue: {e}")
        import mediapipe as mp 
    
    print("✅ LOG: All core libraries are ready.")
except Exception as e:
    print(f"❌ ERROR: Library loading error: {e}")
    sys.exit(1)

# Letter labels (A-Z, skipping J)
harfler = [chr(i) for i in range(65, 91) if i != 74]
model_path = "sign_lang_model.keras"
model = load_model(model_path)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

current_cap_index = 0
cap = cv2.VideoCapture(current_cap_index, cv2.CAP_DSHOW)

def tahmin_et(hand_landmarks):
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    if len(features) == 63:
        prediction = model.predict(np.array([features]), verbose=0)
        return harfler[np.argmax(prediction)]
    return "???"

def kamera_degistir():
    global cap, current_cap_index
    current_cap_index = (current_cap_index + 1) % 3 # Cycle through 0, 1, 2
    print(f"LOG: Camera is changing... New Index: {current_cap_index}")
    cap.release()
    cap = cv2.VideoCapture(current_cap_index, cv2.CAP_DSHOW)

window = tk.Tk()
window.title("Sign Language Recognition System")
window.geometry("850x850")

etiket = tk.Label(window, text="Please Show Your Hand", font=("Arial", 28, "bold"), fg="#2980b9")
etiket.pack(pady=10)

# Camera Change Button
btn_degistir = tk.Button(window, text="Switch Camera (Dahili/iVCam)", command=kamera_degistir, font=("Arial", 12), bg="#e67e22", fg="white", padx=10, pady=5)
btn_degistir.pack(pady=5)

panel = tk.Label(window, bd=2, relief="groove")
panel.pack(padx=10, pady=10)

def kamera_akisi():
    try:
        ret, frame = cap.read()
        if not ret:
            etiket.config(text="Waiting for Camera Connection...", fg="red")
            window.after(500, kamera_akisi)
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        tahmin = "No Hand Detected"
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                tahmin = tahmin_et(handLms)

        etiket.config(text=f"Prediction: {tahmin}", fg="#2980b9")

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        panel.imgtk = imgtk
        panel.configure(image=imgtk)

        window.after(10, kamera_akisi)
    except Exception as e:
        window.after(10, kamera_akisi)

print("🚀 LOG: System Ready!")
kamera_akisi()
window.mainloop()
cap.release()
