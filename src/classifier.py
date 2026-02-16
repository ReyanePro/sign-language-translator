"""
Sign Language Classifier
Loads trained model and predicts ASL letters from hand landmarks.
"""

import numpy as np
import pickle
import json
import os


class SignClassifier:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models",
                "asl_classifier.pkl",
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train_model.py first."
            )

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.encoder = model_data["encoder"]
        self.classes = list(self.encoder.classes_)

        # Confidence history for smoothing
        self.history = []
        self.history_size = 5

    def _normalize(self, landmarks):
        """Normalize landmarks relative to wrist."""
        coords = landmarks.reshape(-1, 2)
        wrist = coords[0]
        centered = coords - wrist
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        if max_dist > 0:
            scaled = centered / max_dist
        else:
            scaled = centered
        return scaled.flatten()

    def _compute_features(self, normalized):
        """Compute feature vector from normalized landmarks."""
        coords = normalized.reshape(-1, 2)
        fingertips = [4, 8, 12, 16, 20]
        finger_bases = [2, 5, 9, 13, 17]

        feat = list(normalized)

        # Fingertip pair distances
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                dist = np.linalg.norm(coords[fingertips[i]] - coords[fingertips[j]])
                feat.append(dist)

        # Fingertip to wrist distances
        for tip in fingertips:
            dist = np.linalg.norm(coords[tip] - coords[0])
            feat.append(dist)

        # Finger curl
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

        return np.array(feat).reshape(1, -1)

    def predict(self, landmarks):
        """
        Predict ASL letter from raw landmarks.
        Returns (letter, confidence) tuple.
        """
        normalized = self._normalize(landmarks)
        features = self._compute_features(normalized)

        # Get probabilities
        proba = self.model.predict_proba(features)[0]
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx]
        letter = self.classes[pred_idx]

        return letter, confidence

    def predict_smooth(self, landmarks, threshold=0.6):
        """
        Predict with temporal smoothing.
        Uses a sliding window of predictions for stability.
        Returns (letter, confidence) or (None, 0) if uncertain.
        """
        letter, confidence = self.predict(landmarks)

        self.history.append((letter, confidence))
        if len(self.history) > self.history_size:
            self.history.pop(0)

        # Count votes in history
        if len(self.history) < 3:
            return letter, confidence

        votes = {}
        total_conf = {}
        for l, c in self.history:
            votes[l] = votes.get(l, 0) + 1
            total_conf[l] = total_conf.get(l, 0) + c

        # Get most voted letter
        best_letter = max(votes, key=votes.get)
        vote_ratio = votes[best_letter] / len(self.history)
        avg_conf = total_conf[best_letter] / votes[best_letter]

        if vote_ratio >= 0.6 and avg_conf >= threshold:
            return best_letter, avg_conf
        else:
            return None, 0

    def get_top_predictions(self, landmarks, top_k=3):
        """Get top-k predictions with probabilities."""
        normalized = self._normalize(landmarks)
        features = self._compute_features(normalized)

        proba = self.model.predict_proba(features)[0]
        top_indices = np.argsort(proba)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((self.classes[idx], proba[idx]))

        return results

    def reset_history(self):
        """Clear prediction history."""
        self.history = []
