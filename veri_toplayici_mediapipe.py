import cv2
import numpy as np
import os
import mediapipe as mp

# Harf etiketleri
class_names = [chr(i) for i in range(65, 91) if i != 74]  # A-Z, J hariç
base_dir = "el_koordinat_verisi"

# Mediapipe ayarları
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Kamera aç
cap = cv2.VideoCapture(0)

etiket = input("📌 Hangi harf için veri toplanacak? (A-Z, J hariç): ").upper()
if etiket not in class_names:
    print("❌ Geçersiz harf.")
    cap.release()
    exit()

print(f"✅ {etiket} harfi için veri toplama başladı. 'q' tuşuyla çıkabilirsin.")

# Veri sayacı
sayac = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Landmarkları al
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])  # x, y, z

            coords_np = np.array(coords)

            # Klasör ve isim oluştur
            klasor = os.path.join(base_dir, etiket)
            os.makedirs(klasor, exist_ok=True)
            dosya_adi = f"{etiket}_{sayac:03d}.npy"
            np.save(os.path.join(klasor, dosya_adi), coords_np)

            # Ekrana çizim
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            sayac += 1

    cv2.putText(frame, f"Harf: {etiket} | Kayit: {sayac-1}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Veri Toplayici - Mediapipe", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"🧾 Toplam {sayac-1} veri kaydedildi.")
cap.release()
cv2.destroyAllWindows()

