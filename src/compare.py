import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import gc
import warnings

warnings.filterwarnings("ignore")

# --- 1. Path and GPU Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# GPU memory growth settings for GTX 1050 2GB
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e: print(f"GPU Configuration Error: {e}")

# --- 2. Parameters ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32 # Safe limit
EPOCHS = 50
LEARNING_RATE = 0.0001
MODELS_TO_COMPARE = ['mobilenetv2', 'resnet50', 'nasnetmobile']

# --- 3. Data Loader ---
def get_generators(preprocess_fn):
    train_datagen = ImageDataGenerator(
        rotation_range=20, horizontal_flip=True, zoom_range=0.1,
        preprocessing_function=preprocess_fn 
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='categorical'
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )
    return train_gen, test_gen

# --- 4. Model Creation ---
def create_model(model_name, num_classes):
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    if model_name == 'mobilenetv2':
        base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'resnet50':
        base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'nasnetmobile':
        base = tf.keras.applications.NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape)

    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=output)

# --- 5. Main Comparison Loop ---
results_list = []

for m_name in MODELS_TO_COMPARE:
    print(f"\n{'-'*30}\nTraining {m_name.upper()} Model...\n{'-'*30}")
    
    tf.keras.backend.clear_session()
    gc.collect()

    if m_name == 'mobilenetv2': pre_fn = tf.keras.applications.mobilenet_v2.preprocess_input
    elif m_name == 'resnet50': pre_fn = tf.keras.applications.resnet50.preprocess_input
    else: pre_fn = tf.keras.applications.mobilenet_v2.preprocess_input # NASNet için uygun

    train_gen, test_gen = get_generators(pre_fn)
    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())

    model = create_model(m_name, num_classes)
    
    # JSON hatasını önlemek için metrics'i sildik
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy')

    # Training
    model.fit(train_gen, epochs=EPOCHS, verbose=1)

    # Evaluation (Metrics calculated cleanly)
    print(f"Analyzing {m_name}...")
    Y_pred = model.predict(test_gen)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_gen.classes

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    results_list.append({
        'Model': m_name,
        'Accuracy': report['accuracy'],
        'F1-Score': report['macro avg']['f1-score'],
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall']
    })

    # Save
    model.save(os.path.join(BASE_DIR, f"best_model_{m_name}.h5"))
    del model
    print(f"Model {m_name} saved and memory cleared.")

# --- 6. Final Table for Paper ---
df_results = pd.DataFrame(results_list)
print("\n" + "="*60 + "\nMODEL COMPARISON TABLE\n" + "="*60)
print(df_results)
df_results.to_csv(os.path.join(BASE_DIR, "final_comparison_results.csv"), index=False)