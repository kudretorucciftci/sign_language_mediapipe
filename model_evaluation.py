import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# Class labels (No J)
class_labels = [chr(i) for i in range(65, 91) if i != 74]

# ✅ Load Data
X = np.load("X_dataset.npy")
y = np.load("y_labels.npy")

# ✅ Load Model
model = load_model("sign_lang_model.keras")

# ✅ Predict
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=1)

# ✅ Performance Report
print("\n📊 Classification Report:\n")
print(classification_report(y, y_pred_classes, target_names=class_labels))

# ✅ Confusion Matrix
cm = confusion_matrix(y, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("🔍 Confusion Matrix")
plt.tight_layout()
plt.show()
