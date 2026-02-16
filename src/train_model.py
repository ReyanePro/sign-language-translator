"""
Train ASL Sign Language Classifier
Trains a Random Forest + optional Neural Network on collected landmark data.
"""

import numpy as np
import os
import sys
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def normalize_landmarks(landmarks):
    """
    Normalize landmarks relative to wrist position and hand scale.
    This makes the model invariant to hand position and size in frame.
    """
    normalized = []
    for sample in landmarks:
        coords = sample.reshape(-1, 2)  # (21, 2)

        # Center on wrist (landmark 0)
        wrist = coords[0]
        centered = coords - wrist

        # Scale by max distance from wrist
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        if max_dist > 0:
            scaled = centered / max_dist
        else:
            scaled = centered

        normalized.append(scaled.flatten())

    return np.array(normalized)


def compute_features(landmarks):
    """
    Compute additional features from landmarks:
    - Distances between fingertips
    - Angles between finger joints
    """
    features = []
    for sample in landmarks:
        coords = sample.reshape(-1, 2)  # (21, 2)

        # Fingertip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
        fingertips = [4, 8, 12, 16, 20]
        finger_bases = [2, 5, 9, 13, 17]

        feat = list(sample)  # Start with raw normalized landmarks

        # Distances between all fingertip pairs
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                dist = np.linalg.norm(coords[fingertips[i]] - coords[fingertips[j]])
                feat.append(dist)

        # Distances from each fingertip to wrist
        for tip in fingertips:
            dist = np.linalg.norm(coords[tip] - coords[0])
            feat.append(dist)

        # Finger curl: distance from tip to base
        for tip, base in zip(fingertips, finger_bases):
            dist = np.linalg.norm(coords[tip] - coords[base])
            feat.append(dist)

        # Angles between consecutive fingers
        for i in range(len(fingertips) - 1):
            v1 = coords[fingertips[i]] - coords[0]
            v2 = coords[fingertips[i + 1]] - coords[0]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            feat.append(angle)

        features.append(feat)

    return np.array(features)


def train():
    """Train the ASL classifier."""
    print("=" * 50)
    print("🧠 Training ASL Sign Language Classifier")
    print("=" * 50)

    # Load data
    data_file = os.path.join(DATA_DIR, "asl_landmarks.npz")
    if not os.path.exists(data_file):
        print("❌ No data found! Run collect_data.py first.")
        return

    data = np.load(data_file, allow_pickle=True)
    landmarks = data["landmarks"]
    labels = data["labels"]

    print(f"📊 Dataset: {len(landmarks)} samples")

    # Check unique labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"📝 Letters: {', '.join(unique_labels)}")
    print(f"📈 Samples per letter: min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}")

    # Normalize landmarks
    print("\n🔄 Normalizing landmarks...")
    norm_landmarks = normalize_landmarks(landmarks)

    # Compute features
    print("🔧 Computing features...")
    features = compute_features(norm_landmarks)
    print(f"   Feature vector size: {features.shape[1]}")

    # Encode labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    print(f"\n📦 Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # Train Random Forest
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc:.4f}")
    print(f"   Test accuracy:  {test_acc:.4f}")

    # Cross-validation
    print("\n🔀 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(rf, features, encoded_labels, cv=5)
    print(f"   CV scores: {cv_scores}")
    print(f"   CV mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Classification report
    y_pred = rf.predict(X_test)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_data = {
        "model": rf,
        "encoder": encoder,
        "feature_size": features.shape[1],
    }

    model_file = os.path.join(MODEL_DIR, "asl_classifier.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model_data, f)

    # Save label mapping
    label_map = {int(i): str(label) for i, label in enumerate(encoder.classes_)}
    with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n✅ Model saved to {model_file}")
    print(f"✅ Label map saved to {os.path.join(MODEL_DIR, 'label_map.json')}")

    # Feature importance (top 10)
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    print("\n🔝 Top 10 most important features:")
    for idx in top_indices:
        print(f"   Feature {idx}: {importances[idx]:.4f}")

    return rf, encoder


if __name__ == "__main__":
    train()
