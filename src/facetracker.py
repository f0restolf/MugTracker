#!/usr/bin/env python3
import cv2
import numpy as np
import pyvirtualcam
from ultralytics import YOLO
import time
import subprocess

# Config
INPUT_DEV = '/dev/video1'
OUTPUT_DEV = '/dev/video10'
MODEL_PATH = 'models/model.pt'
SMOOTHING = 0.12
DEADZONE = 25
DETECT_EVERY = 3

# Force camera to 1080p MJPG before OpenCV touches it
subprocess.run(['v4l2-ctl', '-d', INPUT_DEV, '--set-fmt-video=width=1920,height=1080,pixelformat=MJPG'], check=False)

print("Loading model...")
model = YOLO(MODEL_PATH)

print("Opening camera...")
cap = cv2.VideoCapture(INPUT_DEV)
width, height = 1920, 1080

smooth_cx, smooth_cy = width/2, height/2
last_box = None

print("Starting pipeline...")
with pyvirtualcam.Camera(width=width, height=height, fps=30, device=OUTPUT_DEV) as vcam:
    frame_count = 0
    start = time.time()
    
    while True:
        for _ in range(2):
            cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            continue
        
        if frame_count % DETECT_EVERY == 0:
            results = model(frame, device='cuda', verbose=False)
            
            best_box = None
            best_area = 0
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area = (x2-x1) * (y2-y1)
                    if area > best_area:
                        best_area = area
                        best_box = (x1, y1, x2, y2)
            last_box = best_box
        
        if last_box:
            target_cx = (last_box[0] + last_box[2]) / 2
            target_cy = (last_box[1] + last_box[3]) / 2
            
            dx = abs(target_cx - smooth_cx)
            dy = abs(target_cy - smooth_cy)
            
            if dx > DEADZONE or dy > DEADZONE:
                smooth_cx += SMOOTHING * (target_cx - smooth_cx)
                smooth_cy += SMOOTHING * (target_cy - smooth_cy)
        
        crop_w, crop_h = width//2, height//2
        x1 = int(max(0, min(width - crop_w, smooth_cx - crop_w//2)))
        y1 = int(max(0, min(height - crop_h, smooth_cy - crop_h//2)))
        
        cropped = frame[y1:y1+crop_h, x1:x1+crop_w]
        resized = cv2.resize(cropped, (width, height))
        
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        vcam.send(frame_rgb)
        
        frame_count += 1
        if frame_count % 30 == 0:
            fps = frame_count / (time.time() - start)
            face = "✓" if last_box else "✗"
            print(f"\r[{frame_count}] {fps:.1f} fps | Face: {face}", end="")
