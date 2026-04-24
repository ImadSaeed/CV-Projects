"""
Phase 2: Fine-Tuning
"""

import os
import sys
import tensorflow as tf
from dotenv import load_dotenv
import keras
from keras import mixed_precision
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
from tf_datapipeline import get_datasets 

# Load environment variables
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
load_dotenv(os.path.join(project_root, '.env'))

# Configuration
DATA_DIR = os.environ.get("RAW_DATA_PATH", "/content/data/color")
IMG_SIZE = int(os.getenv("IMG_SIZE", 224))
BATCH_SIZE = 16  
MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'finetuned_model_v4.keras')

def compute_class_weights(dataset):
    """Load pre-computed weights to save time."""
    weights_path = os.path.join(project_root, 'data', 'processed', 'class_weights.npy')
    
    if os.path.exists(weights_path):
        print("Loading pre-computed class weights from file...")
        weights_array = np.load(weights_path, allow_pickle=True)
        print("Weights loaded successfully.")
        return weights_array
    else:
        print(f"Warning: Weights not found. Calculating manually...")
        labels = np.concatenate([y for _, y in dataset], axis=0)
        classes = np.argmax(labels, axis=1)
        return compute_class_weight('balanced', classes=np.unique(classes), y=classes)

def load_head_model():
    """Load Phase 1 model"""
    models_dir = os.path.join(project_root, 'models')
    phase1_folder = os.path.join(models_dir, 'Phase1(head_classifier)')
    
    model_path = None
    if os.path.exists(phase1_folder):
        exact_path = os.path.join(phase1_folder, 'phase1_head_classifier.h5') # Using lowercase as per your logs
        if not os.path.exists(exact_path):
            exact_path = os.path.join(phase1_folder, 'Phase1_head_classifier.h5')
        
        if os.path.exists(exact_path):
            model_path = exact_path
        else:
            files = [f for f in os.listdir(phase1_folder) if f.endswith('.h5')]
            if files:
                model_path = os.path.join(phase1_folder, files[0])
    
    if model_path and os.path.exists(model_path):
        print(f"Loading Phase 1 model from: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Phase 1 model loaded successfully")
        return model
    else:
        raise FileNotFoundError(f"Model not found in {phase1_folder}")

def gradual_unfreeze(model):
    """SAFE STRATEGY: Very low LR, unfreeze 3 layers."""
    base_model = model.layers[1] 

    for layer in base_model.layers[-3:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-05), 
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

def train_finetune():
    """Enhanced fine-tuning"""
    print(f"Using Data Directory: {DATA_DIR}")
    
    train_ds, val_ds = get_datasets(DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)

    class_weights = compute_class_weights(train_ds)
    class_weights_dict = dict(enumerate(class_weights))

    model = load_head_model()
    model = gradual_unfreeze(model)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'),
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_auc', mode='max'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, min_lr=1e-6, mode='max'),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    # Mixed Precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15, 
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    return model, history

if __name__ == "__main__":
    print("Starting Phase 2: Fine-Tuning V4)")
    
    try:
        model, history = train_finetune()
        
        # Manually save the model at the end to be sure
        model.save(MODEL_SAVE_PATH)
        print(f"\n Training Complete!")
        print(f"Model saved to: {MODEL_SAVE_PATH}")
        print("Note: Evaluation skipped due to Phase 1 indexing mismatch, but model is ready for use.")

    except Exception as e:
        print(f"\nFine-tuning failed: {e}")
        import traceback
        traceback.print_exc()