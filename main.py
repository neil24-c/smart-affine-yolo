# SmartAffineYOLO: Real-Time Traffic Management & Emergency Responding System
# Note: This is a conceptual prototype based on the research paper. Hardware integration (e.g., ultrasonic sensors, GSM modules) is represented as placeholders.

import cv2
import numpy as np
import pytesseract
import time
import smtplib
import geopy

# -------------------------- YOLO SETUP -----------------------------
yolo_config = "yolov5s.cfg"  # Replace with your YOLOv5 config file
yolo_weights = "yolov5s.weights"  # Replace with your YOLOv5 trained weights
labels_path = "coco.names"

with open(labels_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(yolo_weights, yolo_config)
layer_names = net.getLayerNames()
out_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# -------------------------- HELPER FUNCTIONS -----------------------------
def send_emergency_alert(plate_number, location):
    print(f"[ALERT] Accident detected for vehicle: {plate_number}")
    print(f"Location: {location}")
    # Placeholders for actual GSM/SMS/Email functionality
    print("Message sent to hospital, police, and emergency contacts.")

def detect_license_plate(image):
    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # OCR (using pytesseract)
    text = pytesseract.image_to_string(edged, config='--psm 8')
    return text.strip()

def apply_affine_transformation(image):
    rows, cols, ch = image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(image, M, (cols, rows))
    return dst

def detect_accident_and_extract_info(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    plate_number = "Unknown"

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "car" or label == "bus" or label == "truck":
                roi = frame[y:y+h, x:x+w]
                roi = apply_affine_transformation(roi)
                plate_number = detect_license_plate(roi)
                location = "12.9716° N, 77.5946° E"  # Replace with real GPS logic
                send_emergency_alert(plate_number, location)

# -------------------------- MAIN LOOP -----------------------------
cap = cv2.VideoCapture("traffic_footage.mp4")  # Replace with live CCTV feed or cam index

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detect_accident_and_extract_info(frame)
    cv2.imshow('SmartAffineYOLO', frame)

    if cv2.waitKey(1) == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
