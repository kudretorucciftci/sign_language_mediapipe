import os
import numpy as np
from tensorflow.keras.utils import to_categorical

veri_yolu = "el_koordinat_verisi"
X = []
y = []

harfler = [chr(i) for i in range(65, 91) if i != 74]  # A-Z, J hariç

for idx, harf in enumerate(harfler):
    klasor_yolu = os.path.join(veri_yolu, harf)
    if not os.path.isdir(klasor_yolu):
        continue

    for dosya_adi in os.listdir(klasor_yolu):
        if dosya_adi.endswith(".npy"):
            dosya_yolu = os.path.join(klasor_yolu, dosya_adi)
            try:
                veri = np.load(dosya_yolu)
                if veri.shape == (63,):  # Beklenen şekil
                    X.append(veri)
                    y.append(idx)
            except Exception as e:
                print(f"Hata: {dosya_yolu} yüklenemedi → {e}")

X = np.array(X)
y = np.array(y)

np.save("X_mediapipe_advanced.npy", X)
np.save("y_mediapipe_advanced.npy", y)

print("✅ Veri başarıyla hazırlandı.")
print("X shape:", X.shape)
print("y shape:", y.shape)
