from ultralytics import YOLO

# Inisialisasi model YOLOv8
model = YOLO('yolov8n.pt')

# Melatih model
model.train(
    data=r'C:\Penyakit Tomat 90%\dataset baru\data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,  # Anda dapat menyesuaikan ukuran batch sesuai kebutuhan
    project=r'C:\Penyakit Tomat 90%\dataset baru',
    name='exp'
)
