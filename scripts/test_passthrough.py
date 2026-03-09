#!/usr/bin/env python3
"""
Phase 1 Test: Camera Passthrough

This tests the basic pipeline without face detection:
1. Capture from camera
2. Write to virtual camera
3. Verify OBS can see it

Run this first to verify your camera and v4l2loopback are working.

Usage:
    python test_passthrough.py
    python test_passthrough.py --input /dev/video2 --output /dev/video10
"""
import argparse
import cv2
import pyvirtualcam
import numpy as np
import time


def main():
    parser = argparse.ArgumentParser(description="Test camera passthrough to virtual camera")
    parser.add_argument('--input', '-i', default='/dev/video0', help='Input camera device')
    parser.add_argument('--output', '-o', default='/dev/video10', help='Virtual camera device')
    parser.add_argument('--width', '-W', type=int, default=1920, help='Frame width')
    parser.add_argument('--height', '-H', type=int, default=1080, help='Frame height')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS')
    args = parser.parse_args()
    
    print("=== Phase 1: Camera Passthrough Test ===\n")
    
    # Open input camera
    print(f"Opening camera: {args.input}")
    cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print(f"✗ Failed to open camera: {args.input}")
        print("\nTroubleshooting:")
        print("  1. List devices: v4l2-ctl --list-devices")
        print("  2. Test with: ffplay -f v4l2 -i", args.input)
        print("  3. Check permissions: groups $USER (should include 'video')")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ Camera: {actual_w}x{actual_h} @ {actual_fps}fps")
    
    # Open virtual camera
    print(f"\nOpening virtual camera: {args.output}")
    try:
        with pyvirtualcam.Camera(
            width=actual_w, 
            height=actual_h, 
            fps=args.fps,
            device=args.output
        ) as vcam:
            print(f"✓ Virtual camera: {vcam.device}")
            print(f"\n=== Passthrough Running ===")
            print("Open OBS -> Add 'Video Capture Device (V4L2)' -> Select 'FaceTrack'")
            print("Press Ctrl+C to stop\n")
            
            frame_count = 0
            start_time = time.time()
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("⚠ Frame read failed, retrying...")
                        continue
                    
                    # Convert BGR (OpenCV) to RGB (pyvirtualcam)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Send to virtual camera
                    vcam.send(frame_rgb)
                    vcam.sleep_until_next_frame()
                    
                    frame_count += 1
                    
                    # Print stats every second
                    if frame_count % args.fps == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"\r[{frame_count:6d}] {fps:.1f} fps", end="")
            
            except KeyboardInterrupt:
                pass
            
            elapsed = time.time() - start_time
            print(f"\n\n✓ Streamed {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    
    except Exception as e:
        print(f"✗ Failed to open virtual camera: {e}")
        print("\nTroubleshooting:")
        print("  1. Load v4l2loopback:")
        print(f"     sudo modprobe v4l2loopback devices=1 video_nr=10 card_label='FaceTrack' exclusive_caps=1 max_buffers=2")
        print(f"  2. Check device: ls -la {args.output}")
    
    finally:
        cap.release()


if __name__ == "__main__":
    main()
