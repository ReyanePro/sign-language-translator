"""
ASL Sign Language Translator - Web Interface (Gradio)
Beautiful web interface for the ASL translator.
"""

import gradio as gr
import cv2
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_tracker import HandTracker
from src.classifier import SignClassifier
from src.spell_engine import SpellEngine


# ── Global State ──────────────────────────────────────
tracker = HandTracker(max_hands=1, detection_confidence=0.7)
classifier = None
speller = SpellEngine(hold_time=0.8, cooldown=0.4)

try:
    classifier = SignClassifier()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False


def process_frame(frame):
    """Process a single webcam frame."""
    if not MODEL_LOADED or frame is None:
        return frame, "Model not loaded. Run train_model.py first.", "", ""

    # MediaPipe expects RGB, Gradio sends RGB
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = tracker.process_frame(bgr_frame)
    landmarks = tracker.extract_landmarks(results)

    # Draw landmarks on frame
    annotated = bgr_frame.copy()
    annotated = tracker.draw_landmarks(annotated, results)

    detected_letter = ""
    confidence_text = ""
    top_preds_text = ""

    if landmarks is not None:
        letter, confidence = classifier.predict_smooth(landmarks)
        top_preds = classifier.get_top_predictions(landmarks, top_k=5)

        spell_result = speller.update(letter, confidence)

        if letter:
            detected_letter = letter
            confidence_text = f"{confidence * 100:.1f}%"

            # Draw big letter on frame
            h, w = annotated.shape[:2]
            cv2.putText(annotated, letter, (w - 120, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 165, 255), 4)

            # Progress bar
            progress = spell_result["hold_progress"]
            bar_w = int(200 * progress)
            cv2.rectangle(annotated, (w - 220, 100), (w - 20, 115), (60, 60, 60), -1)
            color = (0, 220, 100) if spell_result["confirmed"] else (0, 165, 255)
            cv2.rectangle(annotated, (w - 220, 100), (w - 220 + bar_w, 115), color, -1)

            if spell_result["confirmed"]:
                cv2.putText(annotated, "CONFIRMED!", (w - 200, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 100), 2)

            # Top predictions text
            lines = []
            for pred_l, pred_c in top_preds:
                bar = "█" * int(pred_c * 20)
                lines.append(f"{pred_l}  {bar}  {pred_c*100:.1f}%")
            top_preds_text = "\n".join(lines)

        # Bounding box
        bbox = tracker.get_bounding_box(results, annotated.shape)
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
    else:
        speller.update(None, 0)

    # Convert back to RGB for Gradio
    rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return rgb_annotated, detected_letter, confidence_text, speller.get_display_text()


def add_space():
    speller.add_space()
    return speller.get_display_text()


def backspace():
    speller.backspace()
    return speller.get_display_text()


def clear_text():
    speller.clear()
    if classifier:
        classifier.reset_history()
    return ""


# ── Custom CSS ────────────────────────────────────────
custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1200px !important;
}

.main-title {
    text-align: center;
    font-size: 2.5em;
    font-weight: 700;
    background: linear-gradient(135deg, #FF6B35, #F7C948);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #888;
    font-size: 1.1em;
    margin-top: 5px;
}

.detected-letter {
    font-size: 4em !important;
    font-weight: 700 !important;
    text-align: center !important;
    color: #FF6B35 !important;
}

.spelled-text {
    font-size: 1.5em !important;
    font-weight: 500 !important;
    padding: 15px !important;
    background: #1a1a2e !important;
    border-radius: 10px !important;
    color: white !important;
    min-height: 50px !important;
}
"""

# ── Build Interface ───────────────────────────────────
with gr.Blocks(css=custom_css, title="ASL Sign Language Translator", theme=gr.themes.Soft()) as demo:

    gr.HTML("""
        <div>
            <h1 class="main-title">🤟 ASL Sign Language Translator</h1>
            <p class="subtitle">Real-time American Sign Language detection powered by AI</p>
        </div>
    """)

    with gr.Row():
        # Left: Webcam
        with gr.Column(scale=3):
            webcam = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="📷 Camera Feed",
                height=480,
            )

        # Right: Results
        with gr.Column(scale=1):
            detected = gr.Textbox(
                label="🔤 Detected Letter",
                elem_classes=["detected-letter"],
                interactive=False,
                lines=1,
            )

            confidence = gr.Textbox(
                label="📊 Confidence",
                interactive=False,
                lines=1,
            )

            top_predictions = gr.Textbox(
                label="🏆 Top Predictions",
                interactive=False,
                lines=5,
            )

    # Spelled text
    gr.HTML("<h3 style='margin-top: 20px;'>📝 Spelled Text</h3>")
    spelled = gr.Textbox(
        elem_classes=["spelled-text"],
        interactive=False,
        lines=2,
        show_label=False,
    )

    # Controls
    with gr.Row():
        space_btn = gr.Button("␣ Space", variant="secondary", size="lg")
        back_btn = gr.Button("⌫ Delete", variant="secondary", size="lg")
        clear_btn = gr.Button("🗑️ Clear", variant="stop", size="lg")

    # Instructions
    with gr.Accordion("📖 How to use", open=False):
        gr.Markdown("""
        ### Instructions
        1. **Allow camera access** when prompted
        2. **Show ASL hand signs** in front of your camera
        3. **Hold** a sign steady for ~1 second to confirm the letter
        4. Letters will appear in the text box below
        5. Use **Space** to add a space between words
        6. Use **Delete** to remove the last letter

        ### Tips
        - Keep your hand clearly visible and well-lit
        - Hold signs steady for better accuracy
        - The confidence bar shows how sure the model is
        - Letters J and Z require motion and are not supported (static signs only)

        ### Supported Letters
        A B C D E F G H I K L M N O P Q R S T U V W X Y
        """)

    # ── Event Handlers ────────────────────────────────
    webcam.stream(
        fn=process_frame,
        inputs=[webcam],
        outputs=[webcam, detected, confidence, spelled],
    )

    space_btn.click(fn=add_space, outputs=[spelled])
    back_btn.click(fn=backspace, outputs=[spelled])
    clear_btn.click(fn=clear_text, outputs=[spelled])


if __name__ == "__main__":
    print("🌐 Starting web interface...")
    print("   Open http://localhost:7860 in your browser")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
