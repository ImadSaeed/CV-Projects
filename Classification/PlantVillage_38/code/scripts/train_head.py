import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from dotenv import load_dotenv

# 1. Setup Paths
PROJECT_ROOT = "/content/drive/MyDrive/PlantVillage_Project"
sys.path.append(os.path.join(PROJECT_ROOT, 'code/modules'))

from tf_datapipeline import get_datasets

# 2. Load Environment
load_dotenv(os.path.join(PROJECT_ROOT, 'code/.env'))

DATA_DIR = os.getenv("RAW_DATA_PATH")
IMG_SIZE = int(os.getenv("IMG_SIZE", 224))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models/phase1_head_classifier.h5')
EVAL_DIR = os.path.join(PROJECT_ROOT, 'evaluation/tiny_nn')

def build_head_model():
    """Build EfficientNetV2B0 with frozen base and custom head"""
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False, 
        weights='imagenet', 
        input_shape=(IMG_SIZE, IMG_SIZE, 3), 
        pooling='avg'
    )
    base_model.trainable = False 

    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False) 
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(38, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)

def plot_history(history):
    """Save and show training accuracy/loss curves"""
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 6))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='o')
    plt.title('Phase 1: Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='o')
    plt.title('Phase 1: Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'phase1_training_curves.png'))
    plt.show()

def evaluate_model(model, val_ds):
    """Generate Classification Report and Confusion Matrix"""
    print("\n--- Starting Detailed Evaluation ---")
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    y_true = []
    y_pred = []

    # Iterate through validation set to get labels and predictions
    #  Using val_ds.unbatch() is cleaner for gathering results
    for imgs, labels in val_ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Save Classification Report
    report = classification_report(y_true, y_pred, digits=4)
    print("\nClassification Report:\n", report)
    with open(os.path.join(EVAL_DIR, 'phase1_classification_report.txt'), 'w') as f:
        f.write(report)

    # 2. Plot and Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(22, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Phase 1: Confusion Matrix Heatmap')
    plt.ylabel('True Class ID')
    plt.xlabel('Predicted Class ID')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'phase1_confusion_matrix.png'))
    plt.close()
    print(f"Evaluation files saved to: {EVAL_DIR}")

def train_head():
    """Main training loop"""
    # Load data using your custom pipeline
    train_ds, val_ds = get_datasets(DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    
    model = build_head_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss')
    ]

    print(f"\n Phase 1 Training on: {DATA_DIR}")
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=20, 
        callbacks=callbacks
    )
    
    # Run visualization and evaluation after training
    plot_history(history)
    evaluate_model(model, val_ds)
    
    return model, history

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    model, history = train_head()
    
    # Final safety save
    model.save(MODEL_SAVE_PATH)
    print(f"\nPhase 1 Complete. Final Model: {MODEL_SAVE_PATH}")