import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load Model
model = load_model("sign_lang_model.keras")
letters = [chr(i) for i in range(65, 91) if i != 74]  # A-Z except J

# Mediapipe settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera access failed.")
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand detection
    result = hands.process(rgb)
    features = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            # 21 landmarks × 3 coordinates = 63 features
            if len(features) == 63:
                prediction = model.predict(np.array([features]), verbose=0)
                prediction_char = letters[np.argmax(prediction)]
                cv2.putText(frame, f"Prediction: {prediction_char}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Insufficient coordinates", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        cv2.putText(frame, "No hand detected", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Sign Language Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
