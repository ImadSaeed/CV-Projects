"""
Updated LightGBM Training Script for PlantVillage
Optimized for T4 GPU acceleration
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
from IPython.display import Image, display

def load_data():
    """Load preprocessed features"""
    data_path = "/content/drive/MyDrive/PlantVillage_Project/data/processed/efficientnetv2_features.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    data = np.load(data_path)
    X = data['features']
    y = np.argmax(data['labels'], axis=1)
    class_names = data['class_names']

    # Create output directories
    os.makedirs('/content/drive/MyDrive/PlantVillage_Project/models/lightgbm', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/PlantVillage_Project/evaluation/lightgbm', exist_ok=True)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Data loaded successfully:")
    print(f"- Training samples: {X_train.shape[0]}")
    print(f"- Test samples: {X_test.shape[0]}")
    print(f"- Features: {X_train.shape[1]}")
    print(f"- Classes: {len(class_names)}")

    return X_train, X_test, y_train, y_test, class_names

def train_lightgbm():
    """Train LightGBM model with GPU acceleration"""
    X_train, X_test, y_train, y_test, class_names = load_data()

    # GPU-Optimized Parameters
    params = {
        'objective': 'multiclass',
        'num_class': len(class_names),
        'metric': ['multi_logloss', 'multi_error'],
        'boosting_type': 'gbdt',
        
        # --- GPU SETTINGS ---
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'gpu_use_dp': False,       # Use single precision for speed on T4
        'max_bin': 63,             # Optimized for large feature sets (1280 features)
        # ---------------------

        'max_depth': 12,
        'num_leaves': 63,
        'learning_rate': 0.02,
        'class_weight': 'balanced',
        'min_child_samples': 20,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'random_state': 42,
        'verbose': -1,
        'colsample_bytree': 0.9,
        'subsample': 0.9
    }

    # Create Dataset objects
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    # Modern Callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=10)
    ]

    print("\nTraining LightGBM model on GPU...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[valid_set],
        valid_names=['validation'],
        callbacks=callbacks
    )

    # Evaluation
    print("\nEvaluating model...")
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification Report
    report = classification_report(
        y_test,
        y_pred_classes,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print("\nCLASSIFICATION REPORT:")
    print(report)

    # Save report
    report_path = '/content/drive/MyDrive/PlantVillage_Project/evaluation/lightgbm/classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    # Feature Importance Plot
    plt.figure(figsize=(12, 8))
    lgb.plot_importance(model, max_num_features=20)
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    feat_img_path = '/content/drive/MyDrive/PlantVillage_Project/evaluation/lightgbm/feature_importance.png'
    plt.savefig(feat_img_path)
    plt.close()

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_img_path = '/content/drive/MyDrive/PlantVillage_Project/evaluation/lightgbm/confusion_matrix.png'
    plt.savefig(cm_img_path)
    plt.close()

    # Save model
    joblib.dump(model, '/content/drive/MyDrive/PlantVillage_Project/models/lightgbm/lightgbm_model.pkl')
    
    return model

if __name__ == "__main__":
    print("="*80)
    print("LightGBM GPU Training for PlantVillage")
    print("="*80)
    trained_model = train_lightgbm()
    print("\nLightGBM training complete!")