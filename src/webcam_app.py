"""
ASL Sign Language Translator - Webcam Application
Real-time ASL letter detection and word spelling via webcam.
"""

import cv2
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_tracker import HandTracker
from src.classifier import SignClassifier
from src.spell_engine import SpellEngine


# ── UI Colors ──────────────────────────────────────────────
BG_DARK = (30, 30, 30)
BG_PANEL = (45, 45, 45)
ACCENT = (255, 165, 0)       # Orange
ACCENT_GREEN = (0, 220, 100)
ACCENT_RED = (0, 0, 255)
TEXT_WHITE = (255, 255, 255)
TEXT_GRAY = (170, 170, 170)
TEXT_DIM = (100, 100, 100)


def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=10):
    """Draw a rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def draw_progress_ring(img, center, radius, progress, color, thickness=3):
    """Draw a circular progress indicator."""
    angle = int(360 * progress)
    cv2.ellipse(img, center, (radius, radius), -90, 0, angle, color, thickness)
    cv2.ellipse(img, center, (radius, radius), -90, angle, 360, TEXT_DIM, 1)


def draw_confidence_bar(img, x, y, w, h, confidence, label=""):
    """Draw a horizontal confidence bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)

    # Color based on confidence
    if confidence >= 0.8:
        color = ACCENT_GREEN
    elif confidence >= 0.5:
        color = ACCENT
    else:
        color = ACCENT_RED

    bar_w = int(w * confidence)
    cv2.rectangle(img, (x, y), (x + bar_w, y + h), color, -1)

    if label:
        cv2.putText(img, label, (x + 5, y + h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1)


def main():
    print("=" * 50)
    print("🤟 ASL Sign Language Translator")
    print("=" * 50)

    # Initialize components
    tracker = HandTracker(max_hands=1, detection_confidence=0.7)

    try:
        classifier = SignClassifier()
        print("✅ Model loaded")
    except FileNotFoundError:
        print("❌ No trained model found!")
        print("   Run these steps first:")
        print("   1. python src/collect_data.py")
        print("   2. python src/train_model.py")
        return

    speller = SpellEngine(hold_time=1.0, cooldown=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_timer = time.time()
    fps = 0
    frame_count = 0

    print("\nControls:")
    print("  SPACE  → Add space (next word)")
    print("  BACK   → Delete last letter")
    print("  C      → Clear all text")
    print("  Q      → Quit")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Process hand tracking
        results = tracker.process_frame(frame)
        landmarks = tracker.extract_landmarks(results)

        # Draw hand landmarks
        frame = tracker.draw_landmarks(frame, results)

        # ── Top Panel ─────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), BG_DARK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.putText(frame, "ASL TRANSLATOR", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, ACCENT, 2)
        cv2.putText(frame, "Show ASL signs to spell words",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_GRAY, 1)

        # FPS counter
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_timer = time.time()
        cv2.putText(frame, f"{fps} FPS", (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_DIM, 1)

        # ── Prediction Panel (Right Side) ─────────────
        panel_x = w - 250
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (panel_x, 100), (w, 400), BG_PANEL, -1)
        cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)

        if landmarks is not None:
            # Get predictions
            letter, confidence = classifier.predict_smooth(landmarks)
            top_preds = classifier.get_top_predictions(landmarks, top_k=3)

            # Update speller
            spell_result = speller.update(letter, confidence)

            if letter:
                # Big letter display
                letter_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 4)[0]
                letter_x = panel_x + (250 - letter_size[0]) // 2
                cv2.putText(frame, letter, (letter_x, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, ACCENT, 4)

                # Confidence
                conf_text = f"{confidence * 100:.0f}%"
                cv2.putText(frame, conf_text, (panel_x + 90, 235),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_GRAY, 1)

                # Progress ring for letter confirmation
                progress = spell_result["hold_progress"]
                ring_center = (panel_x + 125, 280)
                ring_color = ACCENT_GREEN if spell_result["confirmed"] else ACCENT
                draw_progress_ring(frame, ring_center, 25, progress, ring_color, 3)

                if spell_result["confirmed"]:
                    cv2.putText(frame, "OK!", (panel_x + 105, 285),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_GREEN, 2)
                else:
                    cv2.putText(frame, "Hold", (panel_x + 98, 285),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1)

                # Top predictions
                cv2.putText(frame, "Top predictions:", (panel_x + 15, 330),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_GRAY, 1)
                for i, (pred_letter, pred_conf) in enumerate(top_preds):
                    y_pos = 350 + i * 22
                    draw_confidence_bar(frame, panel_x + 15, y_pos, 180, 16,
                                        pred_conf, f"{pred_letter} {pred_conf*100:.0f}%")

            # Bounding box
            bbox = tracker.get_bounding_box(results, frame.shape)
            if bbox:
                x1, y1, x2, y2 = bbox
                color = ACCENT_GREEN if spell_result.get("confirmed") else ACCENT
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if letter:
                    cv2.putText(frame, f"{letter} ({confidence*100:.0f}%)",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)
        else:
            cv2.putText(frame, "No hand", (panel_x + 60, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_DIM, 1)
            cv2.putText(frame, "detected", (panel_x + 65, 235),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_DIM, 1)
            speller.update(None, 0)

        # ── Text Panel (Bottom) ───────────────────────
        text_panel_y = h - 100
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (0, text_panel_y), (w, h), BG_DARK, -1)
        cv2.addWeighted(overlay3, 0.85, frame, 0.15, 0, frame)

        cv2.putText(frame, "SPELLED TEXT:", (15, text_panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_GRAY, 1)

        display_text = speller.get_display_text()
        if len(display_text) > 50:
            display_text = "..." + display_text[-47:]

        cv2.putText(frame, display_text, (15, text_panel_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_WHITE, 2)

        # Controls hint
        cv2.putText(frame, "SPACE: space | BACKSPACE: delete | C: clear | Q: quit",
                    (15, text_panel_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1)

        # Show frame
        cv2.imshow("ASL Sign Language Translator", frame)

        # Handle key input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            speller.add_space()
        elif key == 8 or key == 127:  # Backspace
            speller.backspace()
        elif key == ord("c"):
            speller.clear()
            classifier.reset_history()

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
    print("\n👋 Bye!")


if __name__ == "__main__":
    main()
