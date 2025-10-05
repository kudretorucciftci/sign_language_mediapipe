import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# Sınıf etiketleri (J harfi yok)
class_labels = [chr(i) for i in range(65, 91) if i != 74]

# ✅ Verileri yükle
X = np.load("X_mediapipe_advanced.npy")
y = np.load("y_mediapipe_advanced.npy")

# ✅ Modeli yükle
model = load_model("mediapipe_model_v4.keras")

# ✅ Tahmin yap
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=1)

# ✅ Performans raporu
print("\n📊 Sınıflandırma Raporu:\n")
print(classification_report(y, y_pred_classes, target_names=class_labels))

# ✅ Karışıklık matrisi
cm = confusion_matrix(y, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("🔍 Karışıklık Matrisi (Confusion Matrix)")
plt.tight_layout()
plt.show()
