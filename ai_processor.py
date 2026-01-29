import cv2
import base64
import json
import numpy as np
import sys
import os
import time
from queue import Queue
from threading import Thread


sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


from ultralytics import YOLO
from rife_wrapper import RIFEInterpolator


USE_EDSR = False


try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    edsr_path = os.path.join(current_dir, "EDSR_x2.pb")
    video_path = os.path.join(current_dir, "0128.mp4")
    yolo_path = "yolov8n.pt"


    if USE_EDSR and not os.path.exists(edsr_path):
        sys.stderr.write(f"Error: Model missing at {edsr_path}\n")
        sys.stderr.flush()
        sys.exit(1)


    if not os.path.exists(video_path):
        sys.stderr.write(f"Error: Video missing at {video_path}\n")
        sys.stderr.flush()
        sys.exit(1)


    sys.stderr.write("Initializing models...\n")
    sys.stderr.flush()
    
    detect_model = YOLO(yolo_path)
    rife = RIFEInterpolator()
    
    if USE_EDSR:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(edsr_path)
        sr.setModel("edsr", 2)
        sys.stderr.write("EDSR super resolution enabled.\n")
        sys.stderr.flush()
    else:
        sys.stderr.write("Using simple bicubic upscaling.\n")
        sys.stderr.flush()
    
    sys.stderr.write("Models loaded successfully.\n")
    sys.stderr.flush()


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.stderr.write(f"Error: Cannot open video {video_path}.\n")
        sys.stderr.flush()
        sys.exit(1)


    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sys.stderr.write(f"Total frames: {total_frames}\n")
    sys.stderr.flush()


    processed_buffer = []
    BUFFER_SIZE = 240
    TARGET_FPS = 48


    class State:
        previous_score = 0.0


    def send_frame(img_bgr, score, current, total):
        _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        packet = {
            "image": jpg_as_text, 
            "score": round(score, 2), 
            "fps": TARGET_FPS,
            "current": current,
            "total": total
        }
        print(f"DATA_START:{json.dumps(packet)}")
        sys.stdout.flush()


    sys.stderr.write("Processing frames...\n")
    sys.stderr.flush()
    
    frame_count = 0
    prev_enhanced = None
    batch_frames = []


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        frame_count += 1
        
        edsr_start = time.time()
        if USE_EDSR:
            enhanced_frame = sr.upsample(frame)
        else:
            enhanced_frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        edsr_time = time.time() - edsr_start
        
        sys.stderr.write(f"[{frame_count}/{total_frames}] EDSR: {edsr_time:.2f}s\n")
        sys.stderr.flush()


        batch_frames.append(enhanced_frame)


        if frame_count % 20 == 0 and batch_frames:
            results = detect_model(batch_frames, verbose=False, conf=0.1, batch=len(batch_frames))
            max_conf = 0.0
            for result in results:
                if len(result.boxes) > 0:
                    max_conf = max(max_conf, float(result.boxes.conf.max()))
            if max_conf > 0:
                State.previous_score = max_conf * 100
            else:
                State.previous_score *= 0.95
            batch_frames = []


        if prev_enhanced is not None:
            rife_start = time.time()
            mid = rife.interpolate(prev_enhanced, enhanced_frame)
            rife_time = time.time() - rife_start
            
            sys.stderr.write(f"[{frame_count}/{total_frames}] RIFE: {rife_time:.2f}s\n")
            sys.stderr.flush()
            
            processed_buffer.append((prev_enhanced.copy(), State.previous_score))
            if mid is not None:
                processed_buffer.append((mid.copy(), State.previous_score))


        prev_enhanced = enhanced_frame


    if prev_enhanced is not None:
        processed_buffer.append((prev_enhanced, State.previous_score))


    cap.release()
    sys.stderr.write(f"Processing complete. Total buffered frames: {len(processed_buffer)}\n")
    sys.stderr.flush()


    sys.stderr.write("Starting playback at target FPS...\n")
    sys.stderr.flush()
    
    frame_delay = 1.0 / TARGET_FPS
    total_output = len(processed_buffer)


    for idx, (frame, score) in enumerate(processed_buffer):
        start_time = time.time()
        send_frame(frame, score, idx + 1, total_output)
        
        elapsed = time.time() - start_time
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


    sys.stderr.write("Video finished.\n")
    sys.stderr.flush()


except Exception as e:
    sys.stderr.write(f"Python Runtime Error: {str(e)}\n")
    sys.stderr.flush()
    sys.exit(1)
