import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path

# ================= AYARLAR =================
IMG_BOYUT = 224
BATCH_SIZE = 32      
EPOCHS = 20
# ===========================================

MEVCUT_DOSYA = Path(__file__).resolve()
PROJE_ANA_DIZIN = MEVCUT_DOSYA.parent.parent
DATA_DIR = PROJE_ANA_DIZIN / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
MODEL_PATH = 'best_skin_model.keras' # Az önce kaydettiğimiz model

def ince_ayar_yap():
    print(f"\n🚀 FINE-TUNING (İNCE AYAR) BAŞLATILIYOR...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: '{MODEL_PATH}' bulunamadı. Önce train.py çalıştırılmalı.")
        return

    # 1. VERİ YÜKLEME (Aynı ayarlar)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_BOYUT, IMG_BOYUT),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,a
        target_size=(IMG_BOYUT, IMG_BOYUT),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # 2. KAYDEDİLMİŞ MODELİ YÜKLE
    print(f"\n--- Model Yükleniyor: {MODEL_PATH} ---")
    model = load_model(MODEL_PATH)
    
    # Modelin içindeki EfficientNet katmanına ulaş
    base_model = model.layers[0] 
    
    # 3. KİLİDİ AÇ (UNFREEZE)
    base_model.trainable = True
    
    # EfficientNetB0 yaklaşık 237 katmanlıdır.
    # İlk 200 katmanı dondurmaya devam et, son 37 katmanı serbest bırak.
    # Bu, modelin temel bilgilerini korurken detayları öğrenmesini sağlar.
    fine_tune_at = 200
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    print(f"Modelin son katmanları (Layer {fine_tune_at}+) eğitime açıldı.")
    print("Çok düşük öğrenme hızı (1e-5) ile hassas eğitim yapılacak.")

    # 4. YENİDEN DERLE (Çok düşük Learning Rate ŞART!)
    # Hızlı öğrenirse bildiklerini unutur (Catastrophic Forgetting).
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. EĞİTİMİ DEVAM ETTİR
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_skin_model_finetuned.keras', monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)
    ]

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # SONUÇLARI GÖSTER
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    print(f"Fine-Tuning Sonrası En İyi Başarı: {max(val_acc):.2f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.legend()
    plt.title('Fine-Tuning Başarısı')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.title('Fine-Tuning Kaybı')
    plt.show()

if __name__ == "__main__":
    ince_ayar_yap()