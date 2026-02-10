import cv2

def kamera_test():
    print("--- KAMERA TESTI BASLATILIYOR ---")
    for i in range(5):  # 0'dan 4'e kadar indeksleri dene
        print(f"Indeks {i} deneniyor...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ KAMERA BULUNDU! Indeks: {i}")
                print(f"Cozunurluk: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return i
            else:
                print(f"❌ Indeks {i} acildi ama goruntu alinamiyor (Baska bir uygulama kullaniyor olabilir).")
            cap.release()
        else:
            print(f"❌ Indeks {i} bagli kamera yok.")
    
    print("\n--- SONUC ---")
    print("Maalesef hicbir kamera erisilebilir degil.")
    return None

if __name__ == "__main__":
    kamera_test()
