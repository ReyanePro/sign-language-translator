"""
Data Collection Script
Capture hand landmarks for each ASL letter via webcam.
Press the letter key on your keyboard to record samples for that letter.
"""

import cv2
import numpy as np
import os
import sys
import json
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hand_tracker import HandTracker


# ASL alphabet (excluding J and Z which require motion)
ASL_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# Number of samples to collect per letter
SAMPLES_PER_LETTER = 50

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def collect_data():
    """Run the data collection interface."""
    os.makedirs(DATA_DIR, exist_ok=True)

    tracker = HandTracker(max_hands=1, detection_confidence=0.7)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    # Load existing data if any
    data_file = os.path.join(DATA_DIR, "asl_landmarks.npz")
    if os.path.exists(data_file):
        existing = np.load(data_file, allow_pickle=True)
        all_landmarks = list(existing["landmarks"])
        all_labels = list(existing["labels"])
        print(f"📂 Loaded {len(all_landmarks)} existing samples")
    else:
        all_landmarks = []
        all_labels = []

    # Count samples per letter
    sample_counts = {}
    for letter in ASL_LETTERS:
        sample_counts[letter] = sum(1 for l in all_labels if l == letter)

    current_letter = None
    recording = False
    record_count = 0
    last_record_time = 0
    record_interval = 0.1  # seconds between auto-captures

    print("\n" + "=" * 50)
    print("🤟 ASL Data Collection")
    print("=" * 50)
    print("Instructions:")
    print("  1. Press a letter key (A-Y) to select that letter")
    print("  2. Press SPACE to start/stop recording")
    print("  3. Show the sign in front of the camera")
    print("  4. Samples are captured automatically while recording")
    print("  5. Press 'S' to save data")
    print("  6. Press 'Q' to quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        results = tracker.process_frame(frame)
        frame = tracker.draw_landmarks(frame, results)

        # Draw UI
        h, w = frame.shape[:2]

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)

        if current_letter:
            status = f"Letter: {current_letter} | Samples: {sample_counts.get(current_letter, 0)}/{SAMPLES_PER_LETTER}"
            color = (0, 255, 0) if recording else (255, 255, 255)
            cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if recording:
                cv2.putText(frame, "● RECORDING", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Press SPACE to record", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "Press a letter key (A-Y) to start", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Progress bar on the right
        bar_x = w - 200
        cv2.rectangle(frame, (bar_x, 90), (w - 10, 90 + len(ASL_LETTERS) * 22 + 10), (30, 30, 30), -1)
        for i, letter in enumerate(ASL_LETTERS):
            count = sample_counts.get(letter, 0)
            y_pos = 105 + i * 22
            ratio = min(count / SAMPLES_PER_LETTER, 1.0)

            # Background bar
            cv2.rectangle(frame, (bar_x + 30, y_pos), (bar_x + 180, y_pos + 15), (60, 60, 60), -1)
            # Progress bar
            bar_color = (0, 255, 0) if ratio >= 1.0 else (0, 165, 255)
            cv2.rectangle(frame, (bar_x + 30, y_pos), (bar_x + 30 + int(150 * ratio), y_pos + 15), bar_color, -1)
            # Letter label
            label_color = (0, 255, 255) if letter == current_letter else (200, 200, 200)
            cv2.putText(frame, letter, (bar_x + 8, y_pos + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)

        # Auto-record when recording is active
        if recording and current_letter:
            current_time = time.time()
            if current_time - last_record_time >= record_interval:
                landmarks = tracker.extract_landmarks(results)
                if landmarks is not None:
                    all_landmarks.append(landmarks)
                    all_labels.append(current_letter)
                    sample_counts[current_letter] = sample_counts.get(current_letter, 0) + 1
                    last_record_time = current_time

                    # Flash effect
                    cv2.rectangle(frame, (0, h - 5), (w, h), (0, 255, 0), -1)

                    if sample_counts[current_letter] >= SAMPLES_PER_LETTER:
                        recording = False
                        print(f"  ✅ {current_letter}: {SAMPLES_PER_LETTER} samples collected!")

        # Bounding box
        bbox = tracker.get_bounding_box(results, frame.shape)
        if bbox:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if recording else (255, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.imshow("ASL Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == 13:
            # Save data
            if all_landmarks:
                np.savez(
                    data_file,
                    landmarks=np.array(all_landmarks),
                    labels=np.array(all_labels),
                )
                print(f"💾 Saved {len(all_landmarks)} samples to {data_file}")
            else:
                print("⚠️  No data to save")
        elif key == ord(" "):
            # Toggle recording
            if current_letter:
                recording = not recording
                if recording:
                    print(f"  🔴 Recording '{current_letter}'...")
                else:
                    print(f"  ⏹️  Stopped recording '{current_letter}'")
        elif chr(key).upper() in ASL_LETTERS if key < 128 else False:
            new_letter = chr(key).upper()
            if new_letter != current_letter:
                recording = False
                current_letter = new_letter
                print(f"\n📌 Selected letter: {current_letter} ({sample_counts.get(current_letter, 0)} samples)")

    # Auto-save on exit
    if all_landmarks:
        np.savez(
            data_file,
            landmarks=np.array(all_landmarks),
            labels=np.array(all_labels),
        )
        print(f"\n💾 Auto-saved {len(all_landmarks)} samples")

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()

    # Print summary
    print("\n📊 Collection Summary:")
    print("-" * 30)
    total = 0
    for letter in ASL_LETTERS:
        count = sample_counts.get(letter, 0)
        total += count
        status = "✅" if count >= SAMPLES_PER_LETTER else "❌"
        print(f"  {letter}: {count:4d} samples {status}")
    print(f"\n  Total: {total} samples")


if __name__ == "__main__":
    collect_data()
