import os
import cv2
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

def apply_clahe_logic(image_np):
    """
    Handles both grayscale and color images
    Expects a Numpy array from the TensorFlow pipeline.
    """
    # Convert to uint8 for OpenCV processing
    img = image_np.astype(np.uint8)
    
    # FIX: Handle grayscale images (1 channel)
    if len(img.shape) == 2:  # Grayscale: (height, width)
        # Convert to RGB by repeating the single channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:  # Grayscale: (height, width, 1)
        # Remove the extra dimension and convert to RGB
        img = cv2.cvtColor(img[:,:,0], cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image (with alpha channel)
        # Remove alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Now the image is guaranteed to have 3 channels (RGB)
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

def image_preprocessing(image, label):
    """Main Function called by the tf.data.Dataset Pipeline"""
    # Get the image size from .env (default to 256 if not set)
    IMG_SIZE = int(os.getenv("IMG_SIZE", 256))
    
    # First resize the image
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Bridge Tensorflow to OpenCV/Numpy for CLAHE processing
    image = tf.numpy_function(
        func=apply_clahe_logic,
        inp=[image],
        Tout=tf.float32
    )
    
    # Manually set the shape (required after using tf.numpy_function)
    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    
    return image, label


if __name__ == "__main__":
    print("\n Running preprocessing test...")

    raw_path = os.getenv("RAW_DATA_PATH")
    first_folder = next(f for f in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, f)))
    
    # Find any image file (not just JPG)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    first_image = next(
        i for i in os.listdir(os.path.join(raw_path, first_folder)) 
        if i.lower().endswith(image_extensions)
    )
    
    test_path = os.path.join(raw_path, first_folder, first_image)

    # Manually Load The Image (Simulating what the TF pipeline does)
    img = cv2.imread(test_path)
    if img is None:
        print(f"Error: Could not load image at {test_path}")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Test the function with dummy label
        processed_img, _ = image_preprocessing(img, "test_label")

        if processed_img is not None:
            print("-" * 40)
            print(f" Success! Image: {first_image}")
            print(f"   - Original shape: {img.shape}")
            print(f"   - Processed shape: {processed_img.shape}")
            print(f"   - Max pixel: {np.max(processed_img.numpy()):.2f}")
            print(f"   - Min pixel: {np.min(processed_img.numpy()):.2f}")
            print(" Preprocessing pipeline is ready!")
        else:
            print("Test failed: Image not processed.")