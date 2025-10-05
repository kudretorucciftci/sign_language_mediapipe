import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Modeli yükle
model = load_model("mediapipe_model_v4.keras")
harfler = [chr(i) for i in range(65, 91) if i != 74]  # J harfi hariç A-Z

# Mediapipe ayarları
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Kamera başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Kamera erişimi başarısız.")
        break

    # Görüntüyü çevir ve RGB'ye dönüştür
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tahmin yapmadan önce el tespiti
    result = hands.process(rgb)
    features = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            # 21 landmark × 3 koordinat = 63 özellik
            if len(features) == 63:
                prediction = model.predict(np.array([features]), verbose=0)
                tahmin = harfler[np.argmax(prediction)]
                cv2.putText(frame, f"Tahmin: {tahmin}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Yetersiz koordinat", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # El bağlantılarını çiz
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        cv2.putText(frame, "El algılanamadı", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Görüntüyü göster
    cv2.imshow("Tahmin", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
