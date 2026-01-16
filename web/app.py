import base64
import cv2
import numpy as np
import time
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "results/finetuned/train/weights/best.pt"
CAPTURES_DIR = Path(__file__).parent.parent / "captures"
DANGEROUS_CLASSES = ["knife", "scissors"]
CONFIDENCE_THRESHOLD = 0.5
CAPTURE_COOLDOWN = 3  # seconds between captures

model = None
last_capture_time = 0


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    CAPTURES_DIR.mkdir(exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    print(f"Model loaded from {MODEL_PATH}")
    yield


app = FastAPI(lifespan=lifespan)


def save_capture(frame, detections):
    """Save frame with bounding boxes drawn."""
    img = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 55, 255) if det["dangerous"] else (55, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {int(det['confidence'] * 100)}%"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    classes = "_".join(set(d["class"] for d in detections))
    filename = f"{timestamp}_{classes}.jpg"
    cv2.imwrite(str(CAPTURES_DIR / filename), img)
    return filename


@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "static/index.html")


@app.get("/captures")
async def captures_page():
    return FileResponse(Path(__file__).parent / "static/captures.html")


@app.get("/api/captures")
async def list_captures():
    """List all captured images."""
    captures = []
    for f in sorted(CAPTURES_DIR.glob("*.jpg"), reverse=True):
        captures.append(
            {
                "filename": f.name,
                "url": f"/captures/{f.name}",
                "timestamp": f.stem.split("_")[0] + "_" + f.stem.split("_")[1],
                "classes": "_".join(f.stem.split("_")[2:]),
            }
        )
    return {"captures": captures}


@app.delete("/api/captures/{filename}")
async def delete_capture(filename: str):
    """Delete a captured image."""
    filepath = CAPTURES_DIR / filename
    if filepath.exists():
        filepath.unlink()
        return {"success": True}
    return {"success": False, "error": "File not found"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()

            # Decode base64 image
            img_data = base64.b64decode(data.split(",")[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Run detection
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    detections.append(
                        {
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": [x1, y1, x2, y2],
                            "dangerous": class_name in DANGEROUS_CLASSES,
                        }
                    )

            # Save capture if dangerous objects detected (with cooldown)
            global last_capture_time
            captured = None
            has_dangerous = any(d["dangerous"] for d in detections)
            if has_dangerous and time.time() - last_capture_time > CAPTURE_COOLDOWN:
                captured = save_capture(frame, detections)
                last_capture_time = time.time()

            await websocket.send_json({"detections": detections, "captured": captured})

    except WebSocketDisconnect:
        print("Client disconnected")


app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")
app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
)
