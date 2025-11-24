import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ================= AYARLAR =================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# VERİ YOLUNU OTOMATİK BUL (Hata riskini sıfırlar)
MEVCUT_DOSYA = Path(__file__).resolve()
PROJE_ANA_DIZIN = MEVCUT_DOSYA.parent.parent # skin-noise-detection klasörü
DATA_DIR = PROJE_ANA_DIZIN / "data"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"

print(f"Eğitim Verisi Yolu: {TRAIN_DIR}")
print(f"Doğrulama Verisi Yolu: {VAL_DIR}")
# ===========================================

def modeli_egit():
    # Klasör kontrolü
    if not TRAIN_DIR.exists():
        print("HATA: Train klasörü bulunamadı! Önce 'prepare.py' çalıştırılmalı.")
        return

    # 1. VERİ YÜKLEME VE ARTIRMA (Data Augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,         # Pikselleri 0-1 arasına çek
        rotation_range=20,      # Resmi rastgele döndür
        width_shift_range=0.1,  # Kaydır
        height_shift_range=0.1,
        horizontal_flip=True,   # Aynala
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    print("\n--- Veriler Yükleniyor ---")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Sınıf isimlerini görelim
    siniflar = list(train_generator.class_indices.keys())
    print(f"Tespit Edilen Sınıflar: {siniflar}")
    num_classes = len(siniflar)

    # 2. MODEL MİMARİSİ (MobileNetV2 - Transfer Learning)
    # Önceden eğitilmiş 'göz' katmanlarını alıyoruz
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False # Hazır bilgileri dondur, bozulmasın

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2), # Ezberlemeyi önle
        layers.Dense(num_classes, activation='softmax') # Çıkış katmanı
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 3. EĞİTİMİ YÖNETME (Callback)
    callbacks = [
        # Val_loss 3 tur iyileşmezse dur
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        # En iyi modeli kaydet
        ModelCheckpoint('en_iyi_model.keras', monitor='val_loss', save_best_only=True)
    ]

    # 4. BAŞLAT
    print("\n--- Eğitim Başlıyor (Bu işlem biraz sürebilir) ---")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # 5. SONUÇLARI GÖRSELLEŞTİR
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Eğitim Başarısı (Train Acc)')
    plt.plot(val_acc, label='Doğrulama Başarısı (Val Acc)')
    plt.legend()
    plt.title('Başarı Grafiği')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Eğitim Kaybı (Train Loss)')
    plt.plot(val_loss, label='Doğrulama Kaybı (Val Loss)')
    plt.legend()
    plt.title('Hata/Kayıp Grafiği')
    
    plt.show()
    print("İşlem Bitti! Model 'en_iyi_model.keras' adıyla kaydedildi.")

if __name__ == "__main__":
    modeli_egit()