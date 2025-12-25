import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Hyperparameters ---
IMG_HEIGHT = 450
IMG_WIDTH = 600
BATCH_SIZE = 32
EPOCHS = 30  
LEARNING_RATE = 0.0001
BASE_MODEL_TRAINABLE = False # Freeze base model for transfer learning at the beginning

# Main path to data directory (adjust according to your folder structure)
DATA_DIR = "data" 

# Filename for saving the model
MODEL_FILENAME = 'best_model_mobilenet_cpu.keras'

# --- 1. Data preprocessing and loading datasets ---

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Data augmentation for TRAIN
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input 
)

# Preprocessing only for VALIDATION and TEST
val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Train generator
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Validation generator
validation_generator = val_test_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'validation'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Test generator
test_generator = val_test_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'test'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Automatically determine number of classes
NUM_CLASSES = len(train_generator.class_indices)
print(f"Total Number of Classes: {NUM_CLASSES}")

# --- 2. Create model (Transfer learning) ---

def create_optimal_model(input_shape, num_classes):
    """MobileNetV2-based model optimized for transfer learning."""
    
    # Load base model (ImageNet weights without top layer)
    base_model = MobileNetV2(
        weights='imagenet',        
        include_top=False,         
        input_shape=input_shape    
    )

    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = BASE_MODEL_TRAINABLE

    # Add new classification layers (Head)
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Flatten feature maps
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) 
    predictions = Dense(num_classes, activation='softmax')(x) 

    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

# Compile the model
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
model = create_optimal_model(input_shape, NUM_CLASSES)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Summary:")
model.summary()

# --- 3. Callbacks and training ---

# Early stopping (prevent overfitting)
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, # Stop if no improvement for 5 epochs
    restore_best_weights=True
)

# Save best model
model_checkpoint = ModelCheckpoint(
    filepath=MODEL_FILENAME,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

print("\nStarting model training (on CPU)...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint]
)

print(f"\nInitial training completed. Best model ({MODEL_FILENAME}) saved.")

# --- 4. Fine-tuning ---

print("\nStarting fine-tuning...")

# Unfreeze frozen layers of base model
model.trainable = True

# Freeze all layers except the last layers (e.g., last 50 layers)
for layer in model.layers[:-50]:
    layer.trainable = False

# Recompile with very low learning rate
FINE_TUNE_LR = LEARNING_RATE / 10
model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS + 10, # Add 10 more epochs on top of initial EPOCHS
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint] # Use same callbacks list
)

# --- 5. Model evaluation ---

# Load best model
final_model = tf.keras.models.load_model(MODEL_FILENAME)

# Evaluate on test data
print("\nEvaluating on test set...")
loss, accuracy = final_model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")