Real-Time Object Detection with YOLOv8 (ONNX)
This project demonstrates real-time object detection using YOLOv8 exported to ONNX format and executed with ONNX Runtime.
________________________________________
 Features
•	Real-time object detection using webcam
•	Image object detection
•	Video object detection
•	Fast inference using ONNX Runtime
•	Lightweight and efficient
________________________________________
 Requirements
Install dependencies:
pip install -r requirements.txt
________________________________________
 How to Run
 Webcam Detection
python webcam_object_detection.py
________________________________________
 Image Detection
python image_object_detection.py
________________________________________
 Video Detection
python video_object_detection.py
________________________________________
 Model
This project uses YOLOv8 exported to ONNX format.
To convert the model:
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx")
________________________________________
 Notes
•	Make sure the ONNX model file (yolov8n.onnx) is in the correct path
•	For GPU acceleration, install:
pip install onnxruntime-gpu
•	Default model path may need to be updated inside the scripts
________________________________________
 Project Structure
.
├── yolov8/
├── models/
├── webcam_object_detection.py
├── image_object_detection.py
├── video_object_detection.py
├── requirements.txt
________________________________________
👨‍💻 Author
Developed by Dheya Alameri 🚀
________________________________________

