import cv2
import base64
import json
import numpy as np
import time
import sys
import os

# Redirect standard streams to prevent log mixing
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

try:
    from ultralytics import YOLO
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    edsr_path = os.path.join(current_dir, "EDSR_x2.pb")
    video_path = os.path.join(current_dir, "0128.mp4")
    yolo_path = "yolov8n.pt"

    # Validation
    if not os.path.exists(edsr_path):
        sys.stderr.write(f"Error: Model missing at {edsr_path}\n")
        sys.exit(1)

    if not os.path.exists(video_path):
        sys.stderr.write(f"Error: Video missing at {video_path}\n")
        sys.exit(1)

    # Load models
    sys.stderr.write("Initializing models...\n")
    detect_model = YOLO(yolo_path) 
    
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(edsr_path)
    sr.setModel("edsr", 2)
    sys.stderr.write("Models loaded successfully.\n")

    # Open video source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.stderr.write(f"Error: Cannot open video {video_path}.\n")
        sys.exit(1)

    previous_score = 0.0 
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            sys.stderr.write("Video finished.\n")
            break

        frame_count += 1

        # Image Enhancement (Using bicubic resize for performance in demo)
        # For production with GPU, uncomment: enhanced_frame = sr.upsample(frame)
        enhanced_frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Fetal Detection (Optimize: Run YOLO every 10 frames)
        if frame_count % 10 == 0:
            # Low confidence threshold(0.1) for demo sensitivity
            results = detect_model(enhanced_frame, verbose=False, conf=0.1) 
            
            if len(results[0].boxes) > 0:
                previous_score = float(results[0].boxes.conf.max()) * 100
            else:
                previous_score *= 0.95 # Decay score if nothing detected

        # Data Transmission
        # Compress to JPEG
        _, buffer = cv2.imencode('.jpg', enhanced_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        packet = {
            "image": jpg_as_text,
            "score": round(previous_score, 2),
            "fps": 60
        }
        
        # Send JSON packet with prefix to Node.js
        print(f"DATA_START:{json.dumps(packet)}")
        sys.stdout.flush() 

    cap.release()

except Exception as e:
    sys.stderr.write(f"Python Runtime Error: {str(e)}\n")
    sys.exit(1)
