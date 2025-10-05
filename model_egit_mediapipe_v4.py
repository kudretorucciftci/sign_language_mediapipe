import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Verileri yükle
X = np.load("X_mediapipe_advanced.npy")
y = np.load("y_mediapipe_advanced.npy")  

# Eğitim ve doğrulama verisi ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Modelin mimarisi (Daha derin, dropout’lu)
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(25, activation='softmax')  # 25 sınıf için
])

# Derleme
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 📌 EarlyStopping ve ModelCheckpoint
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("mediapipe_model_v4.keras", monitor='val_loss', save_best_only=True)

# Eğitimi başlat
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("✅ Gelişmiş model eğitildi ve en iyi hali 'mediapipe_model_v4.keras' olarak kaydedildi.")
