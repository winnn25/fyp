<p align="center">
  <img width="585" height="720"
       src="https://github.com/user-attachments/assets/0b5e9f65-433c-4ef4-936a-994ec139e218" />
</p> 

## Overview
The LRN Sign Detection System is an assistive technology project developed as a Final Year Project (FYP) to aid blind students in navigating university campuses. The system is specifically designed to detect and read aloud Lecturer Room Name (LRN) signs, bridging the gap between physical navigation and digital accessibility. 🔍

## Key Components
- The system utilizes a multi-stage computer vision pipeline to ensure reliable results in real-world environments:
- Sign Detection: Features a custom-trained YOLO (You Only Look Once) model that achieves over 90% accuracy. The model was trained on a proprietary dataset of manually captured and annotated campus images, optimized for real-time performance. ⚡
- OCR Integration: After rigorous testing, the system integrates Tesseract OCR to convert detected sign images into text. This module achieves an accuracy rate of over 80% in reading room names. 📖
- Text-to-Speech (TTS): To ensure accessibility, the processed text is converted into audio feedback, allowing blind users to receive room information through speakers or headphones. 🎧

## Technical Workflow
- Video Capture: The system captures live frames from a video feed.
- Detection Logic: The YOLO model scans frames for specific signage; if detected, it triggers the next phase.
- Optical Character Recognition: Tesseract OCR processes the detected sign area.
- Audio Output: Validated text is converted to speech for the user, with an option to repeat the information as needed.

## Core Achievements
- Custom Dataset: Built from the ground up to reflect the specific lighting and architectural conditions of a university campus.
- Real-World Application: Successfully translated complex machine learning theories into a functional tool that addresses specific user needs for the visually impaired. 🎓
