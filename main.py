import cv2 as cv
from ultralytics import YOLO
from datetime import datetime
import os


def log_detection(class_name, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("logs/detections.txt", "a", encoding="utf-8") as f:
        f.write(
            f'{timestamp} - Phát hiện "{class_name}" (Độ tin cậy: {confidence:.2f})\n'
        )


def draw_detection(frame, name, conf, pos):
    x1, y1, x2, y2 = pos

    is_dangerous = name in DANGEROUS_CLASSES
    color = (0, 55, 255) if is_dangerous else (55, 255, 0)

    label = f"{name} {int(conf * 100)}%"

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_color = (0, 0, 0)
    thickness = 4

    (text_width, text_height), baseline = cv.getTextSize(
        label, font, font_scale, thickness
    )

    cv.rectangle(frame, (x1, y1), (x2, y2), color, 4)

    if y1 < text_height * 4:
        cv.rectangle(
            frame,
            (x2 - text_width - 30, y1),
            (x2, y1 + text_height + baseline + 16),
            color,
            -1,
        )

        cv.putText(
            frame,
            label,
            (x2 - text_width - 14, y1 + text_height + baseline),
            font,
            font_scale,
            font_color,
            thickness,
        )
    else:
        cv.rectangle(
            frame,
            (x1 - 2, y1 - text_height - baseline - 16),
            (x1 + text_width + 32, y1),
            color,
            -1,
        )

        cv.putText(
            frame, label, (x1 + 16, y1 - 16), font, font_scale, font_color, thickness
        )


def play_alert_sound():
    pass


os.makedirs("logs", exist_ok=True)

model = YOLO("results-2/runs/detect/train/weights/best.pt")

cap = cv.VideoCapture(0)

assert cap.isOpened(), "Can not open camera!"

camera_fps = cap.get(cv.CAP_PROP_FPS)

frame_count = 0
model_results = None

DANGEROUS_CLASSES = ["knife", "scissors", "person"]
FRAMES_PER_DETECTION = 3

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    frame = cv.flip(frame, 1)

    if frame_count % FRAMES_PER_DETECTION == 0:
        model_results = model(frame, conf=0.3)

    if model_results:
        for result in model_results:
            boxes = result.boxes
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                draw_detection(
                    frame, name=class_name, conf=confidence, pos=map(int, box.xyxy[0])
                )

    cv.putText(
        frame,
        f"FPS: {camera_fps}",
        (10, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3,
    )

    cv.imshow("Object Detection", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
