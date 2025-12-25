import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configurations
IMG_HEIGHT = 450
IMG_WIDTH = 600
BATCH_SIZE = 64 # Using larger batch size during prediction is faster

# File paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data') 
MODEL_PATH = os.path.join(BASE_DIR, 'best_model_mobilenet_cpu.keras')

# Load model and data
def load_data_and_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Error: Model could not be loaded: {MODEL_PATH}")

    print(f"Loading model: {MODEL_PATH}")
    
    final_model = tf.keras.models.load_model(MODEL_PATH)
    
    # ImageDataGenerator configuration
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Load test generator
    test_generator = test_datagen.flow_from_directory(
        directory=os.path.join(DATA_DIR, 'test'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False 
    )
    
    # Get class labels
    class_labels = list(test_generator.class_indices.keys())
    NUM_CLASSES = len(class_labels)
    
    return final_model, test_generator, class_labels, NUM_CLASSES

# Error analysis function
def analyze_errors(model, generator, class_labels, num_classes):
    print("\nPredicting on test set...")
    Y_pred = model.predict(generator)
    y_pred_classes = np.argmax(Y_pred, axis=1)
    y_true_classes = generator.classes 
    filenames = generator.filenames

    # Find indices of misclassified samples
    error_indices = np.where(y_pred_classes != y_true_classes)[0]
    
    if len(error_indices) == 0:
        print("No misclassifications found in the test set.")
        return

    # Collect error details
    error_details = []
    for i in error_indices:
        # Calculate actual loss (Cross-entropy loss)
        y_true_one_hot = np.zeros(num_classes)
        y_true_one_hot[y_true_classes[i]] = 1.0
        loss_value = -np.sum(y_true_one_hot * np.log(Y_pred[i] + 1e-7)) # Added 1e-7 to prevent log(0) error
        
        error_details.append({
            'index': i,
            'filename': filenames[i],
            'true_label': class_labels[y_true_classes[i]],
            'predicted_label': class_labels[y_pred_classes[i]],
            'loss_value': loss_value,
            'confidence': Y_pred[i][y_pred_classes[i]] * 100 
        })

    # Sort by loss to find biggest errors (highest loss is worst error)
    error_details.sort(key=lambda x: x['loss_value'], reverse=True)

    print(f"\nERROR ANALYSIS: Total misclassifications: {len(error_indices)}")
    print("-" * 60)
    print("HIGHEST LOSS RANKING (Errors where model was confident but wrong)")
    print("-" * 60)

    # Prepare output for console and file
    output_lines = []
    output_lines.append(f"ERROR ANALYSIS: Total misclassifications: {len(error_indices)}")
    output_lines.append("-" * 60)
    output_lines.append("HIGHEST LOSS RANKING (Errors where model was confident but wrong)")
    output_lines.append("-" * 60)

    # Process top 20 worst predictions
    for i, err in enumerate(error_details[:20]):
        console_msg = f"Error {i+1}: Loss={err['loss_value']:.4f}"
        file_msg = f"Error {i+1}: Loss={err['loss_value']:.4f}"
        print(console_msg)
        output_lines.append(file_msg)
        
        console_msg2 = f"  File: {err['filename']}"
        file_msg2 = f"  File: {err['filename']}"
        print(console_msg2)
        output_lines.append(file_msg2)
        
        console_msg3 = f"  True label: {err['true_label']}"
        file_msg3 = f"  True Label: {err['true_label']}"
        print(console_msg3)
        output_lines.append(file_msg3)
        
        console_msg4 = f"  Predicted: {err['predicted_label']} ({err['confidence']:.2f}% confidence)"
        file_msg4 = f"  Predicted: {err['predicted_label']} ({err['confidence']:.2f}% confidence)"
        print(console_msg4)
        output_lines.append(file_msg4)
        
        print("-" * 60)
        output_lines.append("-" * 60)

    # Write errors to file
    output_file = os.path.join(BASE_DIR, 'errors.txt')
    try:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\nError analysis saved to: {output_file}")
    except Exception as e:
        print(f"Error writing to file: {e}")

# --- 3. Main execution block ---
if __name__ == "__main__":
    try:
        model, generator, class_labels, num_classes = load_data_and_model()
        analyze_errors(model, generator, class_labels, num_classes)
        
    except FileNotFoundError as e:
        print(e)
        print("\nPlease run train.py to train and save the model.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")