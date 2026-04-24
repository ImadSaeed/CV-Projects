"""
Tiny Neural Network for PlantVillage Classification
Uses precomputed EfficientNetV2B0 features with class weights for imbalanced data

Dataset Characteristics:
- 54,305 samples
- 1,280 features per sample
- 38 classes with significant imbalance (weights range from 0.26 to 9.40)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def load_data():
    """Load precomputed features and class weights"""
    # Load features
    data = np.load("/content/drive/MyDrive/PlantVillage_Project/data/processed/efficientnetv2_features.npz")
    X = data['features']  # Shape: (54305, 1280)
    y_onehot = data['labels']  # Shape: (54305, 38)
    class_names = data['class_names']

    # Load class weights
    class_weights = np.load("/content/drive/MyDrive/PlantVillage_Project/data/processed/class_weights.npy")
    class_weight_dict = {i: float(weight) for i, weight in enumerate(class_weights)}

    return X, y_onehot, class_names, class_weight_dict

def build_model(input_shape, num_classes):
    """Build a model using the Functional API"""
    inputs = Input(shape=input_shape)

    # First dense block with L2 Regularization
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Second dense block
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="Tiny_Plant_Functional_Net")

    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save to Google Drive directly
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/PlantVillage_Project/tiny_nn_training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save to Google Drive directly
    plt.savefig('/content/drive/MyDrive/PlantVillage_Project/tiny_nn_confusion_matrix.png')
    plt.close()

    return cm

def analyze_class_performance(cm, class_names):
    """Analyze per-class performance from confusion matrix"""
    results = []
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'Class': class_name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': cm[i, :].sum()
        })

    return pd.DataFrame(results).sort_values('F1-Score', ascending=False)

def train_model():
    """Main training function with enhanced monitoring and visualization"""
    # Load data
    X, y, class_names, class_weight_dict = load_data()
    num_classes = len(class_names)

    print(f"Dataset Summary:")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {num_classes}")

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=np.argmax(y, axis=1)
    )

    print(f"\nTrain/Test Split:")
    print(f"   Train samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")

    # Build model
    model = build_model((X_train.shape[1],), num_classes)
    model.summary()

    # Enhanced callbacks
    train_callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            '/content/drive/MyDrive/PlantVillage_Project/models/tiny_nn_best.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir='/content/drive/MyDrive/PlantVillage_Project/logs/tiny_nn')
    ]

    # Train with class weights
    print("\nStarting training with class weights...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=128,
        class_weight=class_weight_dict,
        callbacks=train_callbacks,
        verbose=1
    )

    # Save final model to Google Drive
    model.save('/content/drive/MyDrive/PlantVillage_Project/models/tiny_nn_final.h5')

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nTest Metrics:")
    print(f"   Accuracy: {results[1]:.4f}")
    print(f"   AUC: {results[2]:.4f}")
    print(f"   Precision: {results[3]:.4f}")
    print(f"   Recall: {results[4]:.4f}")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Plot confusion matrix
    cm = plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, digits=4))

    # Class performance analysis
    import pandas as pd
    df_results = analyze_class_performance(cm, class_names)
    print("\nClass Performance (Sorted by F1-Score):")
    print(df_results.to_string(index=False))

    # Save class performance to CSV
    df_results.to_csv('/content/drive/MyDrive/PlantVillage_Project/tiny_nn_class_performance.csv', index=False)

    return model, history

if __name__ == "__main__":
    print("="*80)
    print("PlantVillage Tiny NN Training (Functional API)")
    print("="*80)

    # Ensure directories exist
    os.makedirs('/content/drive/MyDrive/PlantVillage_Project/models', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/PlantVillage_Project/logs', exist_ok=True)

    model, history = train_model()
    print("\n Training complete! All files saved to Google Drive:")
    print("   - Model: /content/drive/MyDrive/PlantVillage_Project/models/tiny_nn_*.h5")
    print("   - Visualizations: /content/drive/MyDrive/PlantVillage_Project/*.png")
    print("   - Class performance: /content/drive/MyDrive/PlantVillage_Project/tiny_nn_class_performance.csv")