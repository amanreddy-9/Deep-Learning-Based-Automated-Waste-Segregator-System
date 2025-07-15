# RoboSeg - Automated Hospital Waste Segregation System

This project is focused on creating a smart system that automatically classifies and separates hospital waste into biodegradable and non-biodegradable categories. The goal is to make hospitals cleaner, safer, and more efficient by reducing human involvement in waste handling.

We used Raspberry Pi 4, IR sensors, a Pi camera, and servo motors, along with deep learning models, to build this system.

---

## 🧠 Project Overview

Hospital waste is often not sorted properly, leading to health risks and environmental pollution. Our system captures images of the waste using a Pi camera, detects it using IR sensors, classifies it using a machine learning model, and then sorts it using a servo motor. The system is fully automated and runs on a Raspberry Pi, making it low-cost and portable.

---

## 📦 Models Used

We have uploaded two different models in this repository. They are stored in separate folders:

### 1. YOLO Model (You Only Look Once)

- Used mainly for object detection.
- Detects where waste objects are in an image.
- Useful for fast detection, but had lower accuracy in our case.

### 2. ResNet-50 Model

- This is the final and preferred model.
- Classifies waste as biodegradable or non-biodegradable.
- High accuracy and works well even when image conditions vary.

---

## 📁 Folder Structure

- `yolo/` → Contains the YOLO model files.
- `resnet50/` → Contains the ResNet-50 classification model.

Each folder has the trained model used in our waste segregation system.

---

## ⚙️ Hardware Components

- Raspberry Pi 4 (main controller)
- Pi Camera (captures waste images)
- IR Sensor (detects waste presence)
- Servo Motor (routes waste)
- Styrofoam box (chassis for the system)

---

## ✅ Testing and Results

- **ResNet-50** gave the best results and was used in the final version.
- System works in **real-time** and runs efficiently on Raspberry Pi.
- Tested for both speed and classification accuracy.

---

## 📚 Paper and Team

This project is based on our paper:  
**RoboSeg: Automated Hospital Waste Segregation System**

**Team Members:**
- Aman Reddy J  
-Ditheswar Sabbu
---

## 📌 Note

The YOLO and ResNet-50 models are uploaded **separately** so that they can be used or tested independently.

---

## 📧 Contact

If you have questions, suggestions, or want to collaborate:

**amanreddyjukonti@gmail.com**,
**dithu2005@gmail.com**
