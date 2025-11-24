import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(data_dir):
    images = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                image = cv2.imread(img_path)
                if image is not None:
                    images.append(image)
                    labels.append(label)
    
    return np.array(images), np.array(labels)

def augment_data(images):
    augmented_images = []
    for image in images:
        # Example augmentation: flipping the image
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)
        # Add more augmentations as needed
    return np.array(augmented_images)

def normalize_data(images):
    return images / 255.0

def split_data(images, labels, test_size=0.2, validation_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=test_size)
    validation_size_adjusted = validation_size / (1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size_adjusted)
    
    return X_train, X_val, X_test, y_train, y_val, y_test