"""
Feature Extraction and Caching Module for PlantVillage Dataset
This script should be run in Google Colab for best performance.

Key Features:
- Uses EfficientNetV2B0 with Global Average Pooling
- Implements CLAHE preprocessing
- Saves compressed features to Google Drive
- Designed for 38-class PlantVillage dataset
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from dotenv import load_dotenv
import time
from tqdm import tqdm

def setup_environment():
    """Set up paths and environment for Colab"""
    project_path = "/content/drive/MyDrive/PlantVillage_Project"
    modules_path = os.path.join(project_path, "code", "modules")

    # Add to Python path
    if modules_path not in sys.path:
        sys.path.append(modules_path)

    # Load environment variables
    load_dotenv(os.path.join(project_path, "code", ".env"))

    return project_path

def create_feature_extractor():
    """Create EfficientNetV2 feature extractor with Global Average Pooling"""
    print("Creating EfficientNetV2B0 feature extractor...")
    base_model = EfficientNetV2B0(
        weights='imagenet',
        include_top=False,
        pooling='avg',  # Global Average Pooling (outputs 1280-dim vector)
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Freeze weights
    return base_model

def extract_and_cache_features(data_dir, cache_dir="/content/data/processed",
                               cache_name="efficientnetv2_features.npz",
                               force_reextract=False, img_size=224, batch_size=32):
    """
    Extract features from images and cache them

    Args:
        data_dir: Directory containing images
        cache_dir: Directory to save cached features
        cache_name: Name for cached file
        force_reextract: If True, re-extract even if cache exists
        img_size: Image size for processing
        batch_size: Batch size for processing

    Returns:
        tuple: (features, labels, class_names)
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name)

    # Check if cache already exists
    if not force_reextract and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        data = np.load(cache_path)
        return data['features'], data['labels'], data['class_names']

    # Setup environment (for Colab)
    try:
        setup_environment()
    except Exception as e:
        print(f"️ Environment setup warning: {e}")
        print("Make sure you're running this in Google Colab with proper paths")

    # Create feature extractor
    feature_extractor = create_feature_extractor()

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )

    # Get class names
    class_names = dataset.class_names
    print(f"Found {len(class_names)} classes: {', '.join(class_names[:5])}...")

    # Extract features
    features = []
    labels = []
    total_images = 0

    print(f" Extracting features (batch_size={batch_size}, img_size={img_size})...")
    for images, batch_labels in tqdm(dataset):
        try:
            # Import here to avoid circular imports
            from pipelines import create_augmented_preprocessing

            # Preprocess images
            processed_images = []
            for image in images:
                processed_img, _ = create_augmented_preprocessing(image, tf.constant(0), is_training=False)
                processed_images.append(processed_img)

            processed_images = tf.stack(processed_images)

            # Extract features
            batch_features = feature_extractor.predict(processed_images)
            features.append(batch_features)
            labels.append(batch_labels.numpy())

            total_images += images.shape[0]

        except Exception as e:
            print(f" Error processing batch: {e}")
            continue

    # Combine all batches
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    print(f" Successfully extracted features from {total_images} images")
    print(f"   Features shape: {features.shape}")
    print(f"   Feature dimension: {features.shape[1]} (1280 for EfficientNetV2B0)")

    # Save to cache with metadata
    print(f" Saving features to {cache_path}")
    np.savez_compressed(
        cache_path,
        features=features,
        labels=labels,
        class_names=class_names,
        img_size=img_size,
        model_name="EfficientNetV2B0",
        feature_dim=features.shape[1],
        preprocessing="CLAHE + EfficientNetV2 preprocessing",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        total_images=total_images
    )

    return features, labels, class_names

if __name__ == "__main__":
    print("""
    This script is designed to run in Google Colab for best performance.
    So just upload this file to Colab and I will run it there.
    """)