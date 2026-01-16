# Real-time Object Detection

A real-time object detection system for detecting knives and scissors using YOLOv11. Features a web interface with live webcam detection and automatic image capture logging.

## Features

- Real-time object detection via webcam
- Web-based interface with FastAPI backend
- Automatic capture of detected objects
- Gallery view for reviewing captured images
- Detects: knives, scissors (highlighted as dangerous)

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Application

```bash
cd web
uvicorn app:app --reload
```

Open http://localhost:8000 in your browser.

- **Detection page**: Live webcam detection
- **Captures page**: View captured images

### Training

```bash
# Prepare dataset (merge knife + scissors datasets)
python preprocessing.py

# Finetune model (open Jupyter notebook)
jupyter notebook yolov11-finetune.ipynb
```

## Project Structure

```
├── web/                       # Web application
│   ├── app.py                # FastAPI backend
│   └── static/               # Frontend HTML
├── dataset/                  # Training datasets
├── main.py                   # Standalone webcam detection (OpenCV)
├── preprocessing.py          # Dataset preparation
├── yolov11-finetune.ipynb   # Model finetuning notebook
└── requirements.txt          # Dependencies
```

## Configuration

Edit `web/app.py` to modify:
- `DANGEROUS_CLASSES`: Classes highlighted in red
- `CONFIDENCE_THRESHOLD`: Detection confidence (default: 0.5)
- `CAPTURE_COOLDOWN`: Seconds between auto-captures (default: 3)
