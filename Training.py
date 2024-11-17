import torch
from ultralytics import YOLO
import cv2
import os

# Path ke model yang telah dilatih
model_path = '/content/runs/detect/train/weights/best.pt'

# Load model YOLOv8
model = YOLO(model_path)

# Path ke dataset untuk melakukan inferensi
dataset_path = 'C:\\Penyakit Tomat 90%\\dataset baru\\valid'  # Sesuaikan dengan path dataset validasi

# Fungsi untuk melakukan inferensi pada gambar
def infer_image(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    return results

# Iterasi melalui dataset dan lakukan inferensi
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(root, file)
            results = infer_image(image_path)
            
            # Proses hasil inferensi (misalnya cetak bounding box dan label)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    label = box.cls
                    confidence = box.conf
                    bbox = box.xyxy
                    print(f'Label: {label}, Confidence: {confidence}, BBox: {bbox}')
                    
            # Opsional: Tampilkan gambar dengan bounding box
            for result in results:
                image_with_boxes = result.plot()  # Menggunakan plot untuk menampilkan bounding box pada gambar
                cv2.imshow('YOLOv8 Detection', image_with_boxes)
                cv2.waitKey(0)  # Tekan tombol apapun untuk lanjut ke gambar berikutnya

cv2.destroyAllWindows()
