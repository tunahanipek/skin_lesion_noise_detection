import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Kütüphane Kurulum Kontrolü:
# Eğer bu dosya çalışırken hata alırsanız, aşağıdaki kütüphaneleri kurduğunuzdan emin olun:
# pip install tensorflow matplotlib scikit-learn

# --- 1. Hiperparametreler ve Yapılandırma ---
IMG_HEIGHT = 450
IMG_WIDTH = 600
BATCH_SIZE = 32
EPOCHS = 30  
LEARNING_RATE = 0.0001
BASE_MODEL_TRAINABLE = False # Transfer öğrenme için temel modeli dondur

# Veri Kümelerinin Ana Yolu (skin-noise-detection/data/..)
DATA_DIR = "data" 

# Model kaydedilirken kullanılacak dosya adı
MODEL_FILENAME = 'best_model_noise_classifier.h5'

# --- 2. Veri Ön İşleme ve Veri Kümelerinin Okunması ---

# MobilNetV2 için özel ön işleme fonksiyonu
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# TRAIN için Data Augmentation (Veri Çeşitliliğini Artırma)
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input 
)

# VALIDATION ve TEST için sadece ön işleme
val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

print("✅ Veri Kümeleri Okunuyor...")
# Train Generator
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Validation Generator
validation_generator = val_test_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'validation'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Test Generator (shuffle=False, Confusion Matrix için kritik)
test_generator = val_test_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'test'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False 
)

# Sınıf sayısını otomatik belirleme
NUM_CLASSES = len(train_generator.class_indices)
class_labels = list(train_generator.class_indices.keys())
print(f"Toplam Sınıf Sayısı: {NUM_CLASSES}")
print(f"Sınıf Etiketleri: {class_labels}")


# --- 3. Optimal CNN Modelini Oluşturma (Transfer Öğrenme) ---

def create_optimal_model(input_shape, num_classes):
    """MobilNetV2 tabanlı, transfer öğrenme için optimize edilmiş model."""
    
    # Temel Modeli Yükle (ImageNet ağırlıkları ve üst katman dahil değil)
    base_model = MobileNetV2(
        weights='imagenet',        
        include_top=False,         
        input_shape=input_shape    
    )

    # Temel Modeli Dondur
    for layer in base_model.layers:
        layer.trainable = BASE_MODEL_TRAINABLE

    # Yeni Sınıflandırma Katmanlarını Ekle (Head)
    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) # Aşırı öğrenmeyi engellemek için
    predictions = Dense(num_classes, activation='softmax')(x) 

    # Modeli Oluştur
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

# Modeli oluştur ve derle
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
model = create_optimal_model(input_shape, NUM_CLASSES)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Özeti (Sınıflandırma Katmanları Eğitilecek):")
model.summary()

# --- 4. Geri Çağrımlar (Callbacks) ve Eğitim ---

# Erken Durdurma
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# En İyi Modeli Kaydetme
model_checkpoint = ModelCheckpoint(
    filepath=MODEL_FILENAME,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

print("\n🚀 Model eğitimi başlıyor (CPU üzerinde)...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint]
)

print(f"\nİlk eğitim tamamlandı. En iyi model ({MODEL_FILENAME}) kaydedildi.")

# --- 5. İnce Ayar (Fine-Tuning) ---

print("\n✨ İnce ayar (Fine-Tuning) başlıyor...")

# Temel modelin dondurulmuş katmanlarının kilidini aç
model.trainable = True

# Son 50 katman dışındaki tüm katmanları dondur
for layer in model.layers[:-50]:
    layer.trainable = False

# Çok düşük bir öğrenme oranı ile yeniden derle
FINE_TUNE_LR = LEARNING_RATE / 10
model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS + 10, # 10 epoch daha fazla eğit
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint]
)


# --- 6. Model Değerlendirme ve Görselleştirme ---

# En iyi modeli yükle
final_model = tf.keras.models.load_model(MODEL_FILENAME)

# Test verisi üzerinde değerlendirme
print("\nTest seti üzerinde değerlendirme...")
loss, accuracy = final_model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)

print(f"Test Kaybı (Loss): {loss:.4f}")
print(f"Test Doğruluğu (Accuracy): {accuracy:.4f}")

# --- 7. Karmaşıklık Matrisi (Confusion Matrix) ---

print("\n📊 Karmaşıklık Matrisi Hesaplanıyor...")

# Test verisi üzerindeki tahminleri al
Y_pred = final_model.predict(test_generator)
y_pred_classes = np.argmax(Y_pred, axis=1) # Tahmin edilen sınıf indeksleri

# Gerçek sınıf indekslerini al
y_true_classes = test_generator.classes 

# Karmaşıklık Matrisini Hesapla
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Karmaşıklık Matrisini Görselleştir
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
plt.title('Karmaşıklık Matrisi (Test Verisi)')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show() # 

# --- 8. Eğitim Geçmişi Grafikleri (Loss ve Accuracy) ---

print("\n📉 Eğitim Geçmişi Grafikleri Çiziliyor...")

# İlk eğitim (history) ve ince ayar (history_fine) geçmişlerini birleştirme
# Tüm epoch'ları tek bir grafikte gösterir
try:
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    epochs_range = range(len(acc))

except NameError:
    # İnce ayar yapılmadıysa sadece history nesnesini kullan
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))


# Grafikler
plt.figure(figsize=(12, 4))

# 1. Doğruluk (Accuracy) Grafiği
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Doğruluğu')
plt.plot(epochs_range, val_acc, label='Doğrulama Doğruluğu')
plt.legend(loc='lower right')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.grid(True)

# 2. Kayıp (Loss) Grafiği
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim Kaybı')
plt.plot(epochs_range, val_loss, label='Doğrulama Kaybı')
plt.legend(loc='upper right')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.grid(True)

plt.show() # 

print("\nProgram sonlandı.")