from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="./dataset/merge.yolov11/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="mps",
    patience=10,  # early stopping
    save=True,
    project="runs/detect",
    name="knife_scissors",
)

# Test
metrics = model.val()
print(metrics.box.map50)
