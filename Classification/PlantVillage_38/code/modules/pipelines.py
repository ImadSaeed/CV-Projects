import os
import cv2
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import albumentations as A
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Load the .env file
load_dotenv()

def apply_clahe_logic(image_np):
    """
    Enhanced version: Handles both single images and batches of images
    """
    # If it's a batch (4D tensor), process each image individually
    if len(image_np.shape) == 4:
        return np.stack([apply_clahe_logic(img) for img in image_np])

    # Rest of your existing code for single images
    img = image_np.astype(np.uint8)

    # Ensure the image has 3 channels
    if len(img.shape) == 2:  # Grayscale: (height, width)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:  # Grayscale: (height, width, 1)
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] != 3:  # Other cases
        raise ValueError(f"Unexpected number of channels in image: {img.shape}")

    # Preprocessing LAB CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img_final = cv2.merge((cl, a, b))
    img_final = cv2.cvtColor(img_final, cv2.COLOR_LAB2RGB)

    # Normalization
    img_normalized = img_final.astype('float32') / 255.0
    return img_normalized

def get_augmentation_pipeline(aug_type='medium'):
    """Get Albumentations augmentation pipeline with different intensities"""
    if aug_type == 'light':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ])
    elif aug_type == 'medium':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.Affine(rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ])
    elif aug_type == 'heavy':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Affine(rotate=(-25, 25), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),  # Fixed var_limit to tuple
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),  # Added fill_value
        ])
    else:
        return A.Compose([])  # No augmentation

def apply_albumentations(image_np, augmentation_pipeline):
    """Apply Albumentations augmentations to a NumPy image or batch of images"""
    # If it's a batch (4D tensor), process each image individually
    if len(image_np.shape) == 4:
        return np.stack([apply_albumentations(img, augmentation_pipeline) for img in image_np])

    if image_np.dtype != np.uint8:
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

    augmented = augmentation_pipeline(image=image_np)
    augmented_image = augmented['image']
    return augmented_image.astype(np.float32) / 255.0

def apply_efficientnet_preprocessing(image_np):
    """
    Apply EfficientNetV2 specific preprocessing
    Handles both single images and batches
    """
    # If it's a batch (4D tensor), process each image individually
    if len(image_np.shape) == 4:
        return np.stack([apply_efficientnet_preprocessing(img) for img in image_np])

    # Convert to uint8 if needed (EfficientNet preprocessing expects 0-255)
    if image_np.dtype != np.uint8:
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)

    # Apply EfficientNetV2 preprocessing
    processed = preprocess_input(image_np)

    # Convert back to float32 in [0,1] range if needed
    if processed.dtype != np.float32:
        processed = processed.astype(np.float32)

    return processed

def preprocess_numpy_image(image_np):
    """Process a single NumPy image (for testing/visualization)"""
    IMG_SIZE = int(os.getenv("IMG_SIZE", 224))
    image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, (IMG_SIZE, IMG_SIZE))

    # Apply CLAHE
    image_processed = tf.numpy_function(
        func=apply_clahe_logic,
        inp=[image_tensor],
        Tout=tf.float32
    )
    image_processed.set_shape((IMG_SIZE, IMG_SIZE, 3))

    # Apply EfficientNet preprocessing
    image_processed = tf.numpy_function(
        func=apply_efficientnet_preprocessing,
        inp=[image_processed],
        Tout=tf.float32
    )
    image_processed.set_shape((IMG_SIZE, IMG_SIZE, 3))

    return image_processed

def create_augmented_preprocessing(image, label, is_training=True, aug_type='medium'):
    """Create preprocessing pipeline with optional augmentation and EfficientNet preprocessing"""
    IMG_SIZE = int(os.getenv("IMG_SIZE", 224))

    # Resize
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    # Apply CLAHE
    image = tf.numpy_function(
        func=apply_clahe_logic,
        inp=[image],
        Tout=tf.float32
    )
    image.set_shape((IMG_SIZE, IMG_SIZE, 3))

    # Apply EfficientNet preprocessing
    image = tf.numpy_function(
        func=apply_efficientnet_preprocessing,
        inp=[image],
        Tout=tf.float32
    )
    image.set_shape((IMG_SIZE, IMG_SIZE, 3))

    # Apply augmentation only for training
    if is_training:
        augmentation_pipeline = get_augmentation_pipeline(aug_type)
        image = tf.numpy_function(
            func=lambda img: apply_albumentations(img, augmentation_pipeline),
            inp=[image],
            Tout=tf.float32
        )
        image.set_shape((IMG_SIZE, IMG_SIZE, 3))

    return image, label

def image_preprocessing(image, label):
    """For tf.data pipeline - expects TensorFlow tensors (legacy, no augmentation)"""
    return create_augmented_preprocessing(image, label, is_training=False)

if __name__ == "__main__":
    print("\nRunning preprocessing test...")

    raw_path = os.getenv("RAW_DATA_PATH")
    if not raw_path or not os.path.exists(raw_path):
        print(f"Error: RAW_DATA_PATH not set or does not exist: {raw_path}")
    else:
        first_folder = next(f for f in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, f)))

        # Find any image file
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        first_image = next(
            i for i in os.listdir(os.path.join(raw_path, first_folder))
            if i.lower().endswith(image_extensions)
        )

        test_path = os.path.join(raw_path, first_folder, first_image)

        # Load image
        img = cv2.imread(test_path)
        if img is None:
            print(f"Error: Could not load image at {test_path}")
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"Original image shape: {img.shape}, dtype: {img.dtype}")

            # Test with augmentation
            augmentation_pipeline = get_augmentation_pipeline('medium')
            processed_img = preprocess_numpy_image(img)
            augmented_img = apply_albumentations(processed_img.numpy(), augmentation_pipeline)

            if augmented_img is not None:
                print("-" * 40)
                print(f"Success! Image: {first_image}")
                print(f"   - Original shape: {img.shape}")
                print(f"   - Processed shape: {processed_img.shape}")
                print(f"   - Augmented shape: {augmented_img.shape}")
                print(f"   - Max pixel: {np.max(augmented_img):.2f}")
                print(f"   - Min pixel: {np.min(augmented_img):.2f}")
                print("Preprocessing and augmentation pipeline with EfficientNetV2 is ready!")
            else:
                print("Test failed: Image not processed.")
