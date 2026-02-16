"""
Hand Tracker Module
Uses MediaPipe to detect and extract hand landmarks in real-time.
"""

import mediapipe as mp
import cv2
import numpy as np


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process_frame(self, frame):
        """Process a BGR frame and return results."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results

    def extract_landmarks(self, results):
        """
        Extract normalized hand landmarks as a flat numpy array.
        Returns array of shape (42,) -> 21 landmarks x 2 (x, y)
        Returns None if no hand detected.
        """
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])

        return np.array(landmarks, dtype=np.float32)

    def extract_landmarks_3d(self, results):
        """
        Extract 3D landmarks as a flat numpy array.
        Returns array of shape (63,) -> 21 landmarks x 3 (x, y, z)
        Returns None if no hand detected.
        """
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        return np.array(landmarks, dtype=np.float32)

    def draw_landmarks(self, frame, results):
        """Draw hand landmarks on the frame."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
        return frame

    def get_bounding_box(self, results, frame_shape):
        """Get bounding box around detected hand."""
        if not results.multi_hand_landmarks:
            return None

        h, w = frame_shape[:2]
        hand_landmarks = results.multi_hand_landmarks[0]

        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        padding = 30
        x_min = max(0, int(min(x_coords)) - padding)
        y_min = max(0, int(min(y_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_max = min(h, int(max(y_coords)) + padding)

        return (x_min, y_min, x_max, y_max)

    def get_handedness(self, results):
        """Return 'Left' or 'Right' for the detected hand."""
        if results.multi_handedness:
            return results.multi_handedness[0].classification[0].label
        return None

    def release(self):
        """Release resources."""
        self.hands.close()
