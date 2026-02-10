import cv2
import numpy as np
import os
import mediapipe as mp

# Class Labels
class_names = [chr(i) for i in range(65, 91) if i != 74]  # A-Z except J
base_dir = "hand_coordinate_data"

# Mediapipe settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open Camera
cap = cv2.VideoCapture(0)

label = input("📌 For which letter will you collect data? (A-Z except J): ").upper()
if label not in class_names:
    print("❌ Invalid letter.")
    cap.release()
    exit()

print(f"✅ Data collection started for letter {label}. Press 'q' to exit.")

# Data Counter
counter = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get Landmarks
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])  # x, y, z

            coords_np = np.array(coords)

            # Create folder and file name
            folder = os.path.join(base_dir, label)
            os.makedirs(folder, exist_ok=True)
            file_name = f"{label}_{counter:03d}.npy"
            np.save(os.path.join(folder, file_name), coords_np)

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            counter += 1

    cv2.putText(frame, f"Letter: {label} | Saved: {counter-1}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Data Collector - Mediapipe", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"🧾 Total {counter-1} samples saved.")
cap.release()
cv2.destroyAllWindows()
