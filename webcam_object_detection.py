import cv2

from yolov8 import YOLOv8

# Gerekli kütüphaneleri içe aktar

# Initialize the webcam  # Webcam'i başlat
cap = cv2.VideoCapture(0)  # Webcam'den video yakala

# Initialize YOLOv7 object detector  # YOLOv8 nesne dedektörünü başlat
model_path = "models/yolov8n.onnx"  # Model yolu
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)  # YOLOv8 dedektörünü başlat

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)  # Tespit edilen nesneler penceresini oluştur
while cap.isOpened():  # Webcam açık olduğu sürece döngü

    # Read frame from the video  # Webcam'den kare oku
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer  # Nesne lokalizatörünü güncelle
    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)  # Tespitleri çiz
    cv2.imshow("Detected Objects", combined_img)  # Görüntüyü göster

    # Press key q to stop  # Durdurmak için q tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
