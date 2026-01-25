"""
Data loading and preprocessing utilities for skin lesion images.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import TRAIN_DIR, IMAGE_ARTIFACTS


def create_data_loaders(image_size=224, batch_size=32):
    """
    Create data loaders with augmentation.
    
    Args:
        image_size: Target image size (assumes square images)
        batch_size: Batch size for training
        
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    
    # Data augmentation for training
    train_augmentation = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        rescale=1./255
    )
    
    # Only rescale for validation and test
    val_augmentation = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_augmentation.flow_from_directory(
        os.path.join(TRAIN_DIR, 'train'),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_augmentation.flow_from_directory(
        os.path.join(TRAIN_DIR, 'validation'),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    test_generator = val_augmentation.flow_from_directory(
        os.path.join(TRAIN_DIR, 'test'),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def get_class_weights(train_generator):
    """
    Calculate class weights to handle imbalanced datasets.
    
    Args:
        train_generator: Training data generator
        
    Returns:
        dict: Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    
    return dict(enumerate(class_weights))
