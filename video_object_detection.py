import cv2
from cap_from_youtube import cap_from_youtube

from yolov8 import YOLOv8

# Gerekli kütüphaneleri içe aktar

# # Initialize video
# cap = cv2.VideoCapture("input.mp4")

videoUrl = 'https://www.youtube.com/watch?v=Hg_rB2hWgko'  # Video URL'si
cap = cap_from_youtube(videoUrl, resolution='720p')  # YouTube'dan video yakala
start_time = 5 # skip first {start_time} seconds  # İlk {start_time} saniyeyi atla
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))  # Başlangıç pozisyonunu ayarla

# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (3840, 2160))  # Çıktı videosunu başlat (yorumlanmış)

# Initialize YOLOv7 model  # YOLOv8 modelini başlat
model_path = "models/yolov8n.onnx"  # Model yolu
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)  # YOLOv8 dedektörünü başlat

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)  # Tespit edilen nesneler penceresini oluştur
while cap.isOpened():  # Video açık olduğu sürece döngü

    # Press key q to stop  # Durdurmak için q tuşuna basın
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video  # Videodan kare oku
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer  # Nesne lokalizatörünü güncelle
    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)  # Tespitleri çiz
    cv2.imshow("Detected Objects", combined_img)  # Görüntüyü göster
    # out.write(combined_img)  # Çıktıya yaz (yorumlanmış)

# out.release()  # Çıktıyı serbest bırak (yorumlanmış)
