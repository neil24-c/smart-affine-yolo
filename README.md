# SmartAffineYOLO

> A Real-Time Traffic Management and Emergency Responding System using YOLOv5 and Affine Transformation

## 🧠 Abstract
Urbanization has increased the number of on-road vehicles, which results in higher accident rates and emergency delays. This project implements an intelligent system that detects accidents in real-time using a combination of **YOLOv5**, **Affine Transformations**, and **ultrasonic sensors**. Upon detecting a crash, the system extracts license plate data, retrieves the owner's emergency information, and sends immediate alerts to hospitals, police, and family members.

## 🚀 Features
- 🔍 **Real-time accident detection** using YOLOv5
- 🧾 **License plate recognition** using OCR
- 🧭 **Emergency notification system** with mock GPS
- 🧠 **Affine transformation** for better accuracy in rotated/skewed plates
- 📊 Lightweight and efficient for live CCTV deployments

## 📂 Project Structure
```
SmartAffineYOLO/
├── main.py                  # Core Python script with detection logic
├── yolov5s.weights          # YOLOv5 weights (add yours here)
├── yolov5s.cfg              # YOLO config file
├── coco.names               # COCO labels
├── traffic_footage.mp4      # Sample video (replace with real-time feed)
├── SmartAffineYOLO_Paper.pdf  # IEEE research paper (optional)
└── README.md
```

## 🛠️ Tech Stack
- Python 3
- OpenCV
- YOLOv5 (Darknet or PyTorch variant)
- Tesseract OCR
- Numpy
- Simulated GPS & alert system

## 🔧 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/SmartAffineYOLO.git
cd SmartAffineYOLO

# Install required packages
pip install opencv-python numpy pytesseract
```

> ⚠️ Note: Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on your system and make sure it’s accessible in PATH.

## 🎥 Demo
Upload a traffic video named `traffic_footage.mp4` or connect a live CCTV feed.
Run the script:
```bash
python main.py
```

## 📈 Results
- Real-time accident detection achieved over **99% accuracy** (based on Euclidean evaluation)
- Response alert time < **3 seconds** from crash detection
- Tested in simulated urban traffic footage with multiple vehicle types

## 📚 Based on Research Paper
> **Title:** SmartAffineYOLO: A System to Perform Real-Time Traffic Management and Emergency Responding  
> **Authors:** Dr. Shantakumar Patil, Dr. Nagashree N., Neil Anthony P S, Suprit U, Samanth A R, et al.  
> **Institution:** Sai Vidya Institute of Technology, Bengaluru

PDF available in repo or upon request.

## 🔮 Future Scope
- Live integration with GSM modules and GPS receivers
- Support for multilingual alerts
- Weather and road condition adaptation
- Integration with emergency services API (e.g., 108 ambulance network)

## 📬 Contact
**Neil Anthony**  
📧 neiltimon2428@gmail.com  
🔗 [GitHub Profile](https://github.com/neil24-c)

---

> “Save lives by responding in time.”
