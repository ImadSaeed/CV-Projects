import os
import tensorflow as tf
import numpy as np
from pipelines import apply_clahe_logic, get_augmentation_pipeline, apply_albumentations
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_augmented_preprocessing(image, label, is_training=True, aug_type='medium'):
    """Preprocessing that handles batches and maps correctly to TF graph"""
    IMG_SIZE = int(os.getenv("IMG_SIZE", 224))
    
    # 1. Resize images
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    # 2. Apply CLAHE (via numpy_function for batch)
    def process_batch(images):
        # images come in as a batch tensor, convert to numpy
        return np.stack([apply_clahe_logic(img) for img in images]).astype(np.float32)

    image = tf.numpy_function(func=process_batch, inp=[image], Tout=tf.float32)
    image.set_shape([None, IMG_SIZE, IMG_SIZE, 3])

    # 3. Apply augmentation only for training
    if is_training:
        augmentation_pipeline = get_augmentation_pipeline(aug_type)
        
        def augment_batch(images):
            # images are already normalized [0,1] from CLAHE, apply_albumentations handles it
            return np.stack([apply_albumentations(img, augmentation_pipeline) for img in images]).astype(np.float32)

        image = tf.numpy_function(func=augment_batch, inp=[image], Tout=tf.float32)
        image.set_shape([None, IMG_SIZE, IMG_SIZE, 3])

    return image, label

def get_datasets(data_dir, batch_size=32, img_size=224):
    """
    Properly splits data into 80% train and 20% validation
    """
    print(f"Creating datasets from: {data_dir}")
    
    # Load Training Subset
    train_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )

    # Load Validation Subset
    val_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )

    # Apply the complex preprocessing (CLAHE + Augmentation)
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_raw.map(
        lambda x, y: create_augmented_preprocessing(x, y, is_training=True),
        num_parallel_calls=AUTOTUNE
    )
    
    val_ds = val_raw.map(
        lambda x, y: create_augmented_preprocessing(x, y, is_training=False),
        num_parallel_calls=AUTOTUNE
    )

    # Optimize for performance
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds
