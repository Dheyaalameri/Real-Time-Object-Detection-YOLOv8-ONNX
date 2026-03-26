import cv2
from imread_from_url import imread_from_url

from yolov8 import YOLOv8

# Gerekli kütüphaneleri içe aktar

# Initialize yolov8 object detector  # YOLOv8 nesne dedektörünü başlat
model_path = "models/yolov8n.onnx"  # Model yolu
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)  # YOLOv8 dedektörünü başlat

# Read image  # Görüntüyü oku
img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"  # Görüntü URL'si
img = imread_from_url(img_url)  # URL'den görüntü oku

# Detect Objects  # Nesneleri tespit et
boxes, scores, class_ids = yolov8_detector(img)

# Draw detections  # Tespitleri çiz
combined_img = yolov8_detector.draw_detections(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)  # Tespit edilen nesneler penceresini oluştur
cv2.imshow("Detected Objects", combined_img)  # Görüntüyü göster
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)  # Görüntüyü kaydet
cv2.waitKey(0)  # Bir tuşa basılmasını bekle
