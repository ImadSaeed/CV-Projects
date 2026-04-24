"""
    Ensemble model :-(lightGBM + Tiny NN)
"""
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Run ensemble inference")
    parser.add_argument("--tiny-model", required=True, help="Path to Tiny NN .h5 model")
    parser.add_argument("--lgbm-model", required=True, help="Path to LightGBM model file (.txt)")
    parser.add_argument("--features", required=True, help="Path to cached features .npz file")
    parser.add_argument("--output-dir", default="results", help="Directory to save ensemble outputs")
    parser.add_argument("--tiny-weight", type=float, default=0.75, help="Weight for Tiny NN (0-1)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Create timestamped subfolder for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"ensemble_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Ensemble results will be saved to: {output_dir}")

    # Load models
    print("Loading Tiny NN model...")
    tiny_model = tf.keras.models.load_model(args.tiny_model)

    print("Loading LightGBM model...")
    lgbm_model = lgb.Booster(model_file=args.lgbm_model)

    # Load cached test features and labels
    print("Loading cached features...")
    data = np.load(args.features)
    X_test = data['features']      # shape: (n_samples, 1280)
    y_test = data['labels']        # shape: (n_samples,)
    print(f"Loaded {X_test.shape[0]} test samples, features shape: {X_test.shape}")

    # Get probabilities
    print("Predicting with Tiny NN...")
    tiny_probs = tiny_model.predict(X_test, batch_size=64, verbose=1)

    print("Predicting with LightGBM...")
    lgbm_probs = lgbm_model.predict(X_test)

    # Weighted soft voting
    print(f"Computing ensemble (Tiny NN weight = {args.tiny_weight})...")
    w_tiny = args.tiny_weight
    ensemble_probs = (w_tiny * tiny_probs) + ((1 - w_tiny) * lgbm_probs)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    # Evaluation
    acc = accuracy_score(y_test, ensemble_preds)
    print(f"\nEnsemble Accuracy: {acc:.4f}")

    # Now, use numeric labels
    report = classification_report(y_test, ensemble_preds, digits=4, zero_division=0)
    print("\nClassification Report:\n", report)

    # Save everything
    pd.DataFrame({
        'true_label': y_test,
        'ensemble_pred': ensemble_preds,
        'max_prob': np.max(ensemble_probs, axis=1)
    }).to_csv(os.path.join(output_dir, "ensemble_predictions.csv"), index=False)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    np.save(os.path.join(output_dir, "ensemble_probs.npy"), ensemble_probs)

    print(f"\nEnsemble complete! All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()