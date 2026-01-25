import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications import EfficientNetV2B0, MobileNetV2, ResNet50, NASNetMobile
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Hyperparameters ---
IMG_HEIGHT = 450
IMG_WIDTH = 600
BATCH_SIZE = 32
EPOCHS = 30  
LEARNING_RATE = 0.0001
BASE_MODEL_TRAINABLE = False # Freeze base model for transfer learning at the beginning
DEFAULT_ARCHITECTURE = 'efficientnetv2b0'  # Primary architecture

# Get the script directory and set relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data") 
METADATA_FILE = os.path.join(PROJECT_DIR, "data", "metadata.csv")

# --- 1. Load metadata and prepare data ---
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # --- DÜZELTME 1: Windows Ters Bölülerini Düzelt ---
    # Python'un anlayacağı evrensel formata çeviriyoruz
    df['filename'] = df['filename'].apply(lambda x: x.replace('\\', '/'))
    
    # --- DÜZELTME 2: Göreceli Yolları Mutlak Yollara Çevir ---
    # ./data/train/... şeklindeki yolları PROJECT_DIR'den başlayacak şekilde çevir
    df['filename'] = df['filename'].apply(lambda x: os.path.join(PROJECT_DIR, "data", x.split('/')[-2], x.split('/')[-1]))
    
    # --- DÜZELTME 3: 'Clean' ve Boş Sınıfları At ---
    # LLM çalıştırmadığın için şimdilik sadece klasörü olanları hedefle.
    # 'clean' sınıfını eğitime ASLA katma. Hepsi 0 ise clean demektir.
    target_cols = ['blurry', 'hairy', 'bubble', 'ruler'] 
    
    # Sadece hedeflediğimiz sütunların olduğu yeni bir liste yap
    print(f"Eğitilecek sınıflar: {target_cols}")
    
    return df, target_cols

def create_dataset(df, artifact_classes, batch_size=BATCH_SIZE, shuffle=True):
    """Create TensorFlow dataset from dataframe"""
    # Extract image paths and labels
    image_paths = df['filename'].values
    labels = df[artifact_classes].values.astype(np.float32)
    
    # Create dataset
    def load_image_and_label(path, label):
        img = tf.py_function(
            lambda p: load_and_preprocess_image(p.numpy().decode('utf-8')),
            [path],
            tf.float32
        )
        img.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

# --- 2. Create model (Transfer learning) ---

def create_efficientnetv2b0_model(input_shape, num_classes):
    """EfficientNetV2B0 model - PRIMARY ARCHITECTURE"""
    base_model = EfficientNetV2B0(
        weights='imagenet',        
        include_top=False,         
        input_shape=input_shape    
    )
    for layer in base_model.layers:
        layer.trainable = BASE_MODEL_TRAINABLE
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) 
    predictions = Dense(num_classes, activation='sigmoid')(x) 
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def create_mobilenetv2_model(input_shape, num_classes):
    """MobileNetV2 model (alternative)"""
    base_model = MobileNetV2(
        weights='imagenet',        
        include_top=False,         
        input_shape=input_shape    
    )
    for layer in base_model.layers:
        layer.trainable = BASE_MODEL_TRAINABLE
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x) 
    predictions = Dense(num_classes, activation='sigmoid')(x) 
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def create_resnet50_model(input_shape, num_classes):
    """ResNet50 model (alternative)"""
    base_model = ResNet50(
        weights='imagenet',        
        include_top=False,         
        input_shape=input_shape    
    )
    for layer in base_model.layers:
        layer.trainable = BASE_MODEL_TRAINABLE
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) 
    predictions = Dense(num_classes, activation='sigmoid')(x) 
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def create_nasnetmobile_model(input_shape, num_classes):
    """NASNetMobile model (alternative)"""
    base_model = NASNetMobile(
        weights='imagenet',        
        include_top=False,         
        input_shape=input_shape    
    )
    for layer in base_model.layers:
        layer.trainable = BASE_MODEL_TRAINABLE
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x) 
    predictions = Dense(num_classes, activation='sigmoid')(x) 
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def get_model(architecture=DEFAULT_ARCHITECTURE, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=4):
    """
    Factory function to create model.
    
    Args:
        architecture: 'efficientnetv2b0' (primary), 'mobilenetv2', 'resnet50', 'nasnetmobile'
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        keras.Model: Compiled model
    """
    
    architectures = {
        'efficientnetv2b0': create_efficientnetv2b0_model,
        'mobilenetv2': create_mobilenetv2_model,
        'resnet50': create_resnet50_model,
        'nasnetmobile': create_nasnetmobile_model
    }
    
    if architecture.lower() not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(architectures.keys())}")
    
    print(f"Creating {architecture} model...")
    model = architectures[architecture.lower()](input_shape, num_classes)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(architecture=DEFAULT_ARCHITECTURE):
    """
    Main training function supporting multiple architectures.
    
    Args:
        architecture: Model architecture to use (default: efficientnetv2b0)
    """
    # Load and prepare data
    metadata_df, artifact_classes = load_data(METADATA_FILE)
    print(f"Loaded metadata with {len(metadata_df)} images")
    print(f"Target artifact classes: {artifact_classes}")
    
    # Split data
    train_df, temp_df = train_test_split(metadata_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = create_dataset(train_df, artifact_classes, shuffle=True)
    val_dataset = create_dataset(val_df, artifact_classes, shuffle=False)
    test_dataset = create_dataset(test_df, artifact_classes, shuffle=False)
    
    # Create model
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    num_artifact_classes = len(artifact_classes)
    model = get_model(architecture, input_shape, num_artifact_classes)
    
    print("Model Summary:")
    model.summary()
    
    # Setup model saving
    model_dir = os.path.join(PROJECT_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_filename = os.path.join(model_dir, f'best_model_{architecture}.keras')
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5,
        restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        filepath=model_filename,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    # Initial training
    print(f"\nStarting {architecture} training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[early_stopping, model_checkpoint]
    )
    print(f"Initial training completed. Best model saved to {model_filename}")
    
    # Fine-tuning
    print("\nStarting fine-tuning...")
    model.trainable = True
    for layer in model.layers[:-50]:
        layer.trainable = False
    
    FINE_TUNE_LR = LEARNING_RATE / 10
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history_fine = model.fit(
        train_dataset,
        epochs=EPOCHS + 10,
        initial_epoch=len(history.history['loss']),
        validation_data=val_dataset,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Evaluation
    final_model = tf.keras.models.load_model(model_filename)
    print("\nEvaluating on test set...")
    loss, accuracy = final_model.evaluate(test_dataset)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot results
    plot_training_history(history, history_fine, output_dir=os.path.join(PROJECT_DIR, 'visualizations'))


def plot_training_history(history_initial, history_fine, output_dir='visualizations'):
    """Plot training and validation loss and accuracy curves"""
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Combine histories
    initial_loss = history_initial.history['loss']
    initial_val_loss = history_initial.history['val_loss']
    initial_accuracy = history_initial.history['accuracy']
    initial_val_accuracy = history_initial.history['val_accuracy']
    
    fine_loss = history_fine.history['loss']
    fine_val_loss = history_fine.history['val_loss']
    fine_accuracy = history_fine.history['accuracy']
    fine_val_accuracy = history_fine.history['val_accuracy']
    
    # Combine epochs for continuous x-axis
    initial_epochs = len(initial_loss)
    total_epochs = initial_epochs + len(fine_loss)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    epochs_initial = range(1, initial_epochs + 1)
    epochs_fine = range(initial_epochs + 1, total_epochs + 1)
    
    axes[0].plot(epochs_initial, initial_loss, 'b-', label='Training Loss (Initial)', linewidth=2)
    axes[0].plot(epochs_initial, initial_val_loss, 'r-', label='Validation Loss (Initial)', linewidth=2)
    axes[0].plot(epochs_fine, fine_loss, 'b--', label='Training Loss (Fine-tune)', linewidth=2)
    axes[0].plot(epochs_fine, fine_val_loss, 'r--', label='Validation Loss (Fine-tune)', linewidth=2)
    axes[0].axvline(x=initial_epochs, color='gray', linestyle=':', alpha=0.7, label='Fine-tuning Start')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[1].plot(epochs_initial, initial_accuracy, 'b-', label='Training Accuracy (Initial)', linewidth=2)
    axes[1].plot(epochs_initial, initial_val_accuracy, 'r-', label='Validation Accuracy (Initial)', linewidth=2)
    axes[1].plot(epochs_fine, fine_accuracy, 'b--', label='Training Accuracy (Fine-tune)', linewidth=2)
    axes[1].plot(epochs_fine, fine_val_accuracy, 'r--', label='Validation Accuracy (Fine-tune)', linewidth=2)
    axes[1].axvline(x=initial_epochs, color='gray', linestyle=':', alpha=0.7, label='Fine-tuning Start')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Train with primary architecture (EfficientNetV2B0)
    train_model(DEFAULT_ARCHITECTURE)
    
    # Optionally train with other architectures:
    # train_model('mobilenetv2')
    # train_model('resnet50')
    # train_model('nasnetmobile')
