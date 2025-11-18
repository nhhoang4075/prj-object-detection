import cv2
from ultralytics import YOLO
from datetime import datetime
import os

if not os.path.exists("logs"):
    os.makedirs("logs")

model = YOLO("yolov8n.pt")

DANGEROUS_CLASSES = ["knife", "scissors", "person"]

cap = cv2.VideoCapture(0)


def log_detection(class_name, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("logs/detections.txt", "a", encoding="utf-8") as f:
        f.write(
            f"{timestamp} - Phát hiện {class_name} (Độ tin cậy: {confidence:.2f})\n"
        )


def play_alert_sound():
    pass


while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.5)  # confidence threshold = 0.5

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            is_dangerous = class_name in DANGEROUS_CLASSES

            if is_dangerous:
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                label = f"DANGEROUS! {class_name} {confidence:.2f}"
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

                log_detection(class_name, confidence)
            else:
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cv2.putText(
        frame,
        f"FPS: {fps}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
