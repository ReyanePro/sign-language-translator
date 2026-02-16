# ASL Sign Language Translator

> Real-time American Sign Language detection and word spelling powered by Computer Vision and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## About

This project translates **American Sign Language (ASL) hand signs** into text in real-time using your webcam. It detects hand landmarks using MediaPipe, classifies them into ASL letters using a trained ML model, and allows you to **spell out words** letter by letter.

### Features

- **Real-time hand detection** using MediaPipe
- **ASL alphabet recognition** (A-Y, excluding J & Z which require motion)
- **Word spelling mode** — hold a sign to confirm a letter, build words in real-time
- **Confidence scoring** — see how sure the model is about each prediction
- **Two interfaces** — OpenCV webcam app + Gradio web interface
- **Runs locally** — no internet needed, all processing on your machine

##  Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Webcam     │───▶│  MediaPipe   │───▶│  Classifier  │───▶│   Spell      │
│   Feed       │    │  Hand Track  │    │  (Random     │    │   Engine     │
│              │    │  21 Landmarks│    │   Forest)    │    │   A→AB→ABC   │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                          │                    │                     │
                          ▼                    ▼                     ▼
                    Hand Landmarks      Letter + Confidence    Spelled Text
```

## Quick Start

### Prerequisites

- Python 3.9+
- Webcam
- macOS / Linux / Windows

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/sign-language-translator.git
cd sign-language-translator

# Run setup (macOS/Linux)
chmod +x setup.sh
./setup.sh

# Or manual setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1: Collect Training Data

Record your own hand signs via webcam:

```bash
python src/collect_data.py
```

**Controls:**
- Press a **letter key** (A-Y) to select that letter
- Press **SPACE** to start/stop recording
- Show the ASL sign in front of the camera
- Aim for **50+ samples** per letter
- Press **S** to save, **Q** to quit

>  **Tip:** Vary your hand position, angle, and distance slightly between samples for better model generalization.

### Step 2: Train the Model

```bash
python src/train_model.py
```

This trains a Random Forest classifier on your collected data and saves the model.

### Step 3: Run the Translator

**Option A — Webcam app (OpenCV):**
```bash
python src/webcam_app.py
```

**Option B — Web interface (Gradio):**
```bash
python web/gradio_app.py
# Open http://localhost:7860
```

##  How to Use

1. Show an ASL hand sign in front of your camera
2. The detected letter appears on screen with a confidence score
3. **Hold the sign steady** for ~1 second to confirm the letter
4. The letter is added to the text
5. Use **SPACE** to add a space between words
6. Use **BACKSPACE** to delete the last letter

##  Project Structure

```
sign-language-translator/
├── src/
│   ├── hand_tracker.py      # MediaPipe hand detection wrapper
│   ├── classifier.py        # ML classification module
│   ├── spell_engine.py      # Word building logic
│   ├── collect_data.py      # Data collection tool
│   ├── train_model.py       # Model training script
│   └── webcam_app.py        # OpenCV webcam application
├── web/
│   └── gradio_app.py        # Gradio web interface
├── models/                  # Trained models (generated)
├── data/                    # Collected data (generated)
├── demo/                    # Screenshots & demo videos
├── requirements.txt
├── setup.sh
└── README.md
```

##  Technologies

| Technology | Purpose |
|---|---|
| **Python** | Core language |
| **MediaPipe** | Hand landmark detection (21 keypoints) |
| **OpenCV** | Video capture & real-time display |
| **scikit-learn** | Random Forest classifier |
| **NumPy** | Data processing & feature engineering |
| **Gradio** | Web interface |

##  Model Details

**Input:** 42 hand landmark coordinates (21 points × 2D) + engineered features:
- Fingertip-to-fingertip distances (10 pairs)
- Fingertip-to-wrist distances (5)
- Finger curl distances (5)
- Inter-finger angles (4)

**Total features:** 66

**Model:** Random Forest (200 trees) with temporal smoothing for prediction stability.

**Normalization:** Landmarks are centered on the wrist and scaled by hand size, making the model invariant to hand position and distance from camera.

##  Contributing

Contributions are welcome! Some ideas:

- [ ] Add LSF (French Sign Language) support
- [ ] Add motion-based signs (J, Z)
- [ ] Neural network model (CNN/LSTM)
- [ ] Text-to-speech output
- [ ] Two-hand sign support

##  License

MIT License — see [LICENSE](LICENSE) for details.

##  Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google for hand tracking
- [ASL Alphabet](https://www.handspeak.com/word/asl-eng/) reference
- The Deaf community for inspiring this project

---

<p align="center">
  Made with ❤️ for accessibility and inclusion
</p>
