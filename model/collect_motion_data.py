import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import RunningMode
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Collect motion-sign sequences for J and Z.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("motion_data"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hand-model", type=Path, default=Path(__file__).with_name("hand_landmarker.task"))
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--padding", type=float, default=0.35)
    return parser.parse_args()


def crop_hand(frame, landmarks, padding):
    height, width = frame.shape[:2]
    x_values = [point.x for point in landmarks]
    y_values = [point.y for point in landmarks]
    left, right = int(min(x_values) * width), int(max(x_values) * width)
    top, bottom = int(min(y_values) * height), int(max(y_values) * height)
    center_x, center_y = (left + right) // 2, (top + bottom) // 2
    side = max(right - left, bottom - top)
    side = max(1, int(side * (1 + padding)))
    left = max(0, center_x - side // 2)
    top = max(0, center_y - side // 2)
    right = min(width, left + side)
    bottom = min(height, top + side)
    left = max(0, right - side)
    top = max(0, bottom - side)
    crop = frame[top:bottom, left:right]
    return cv2.resize(crop, (224, 224)) if crop.size else None


def main():
    args = parse_args()
    if not args.hand_model.is_file():
        raise FileNotFoundError(f"Missing hand model: {args.hand_model}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")
    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.hand_model)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )
    label = "J"
    sequence = []
    recording = False
    timestamp = 0
    capture_message = ""
    capture_message_until = 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a camera frame")
            display = cv2.flip(frame, 1)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                timestamp,
            )
            timestamp += 33
            crop = None
            if result.hand_landmarks:
                crop = crop_hand(frame, result.hand_landmarks[0], args.padding)
                if result.handedness[0][0].category_name == "Left":
                    crop = cv2.flip(crop, 1)
            if recording and crop is not None and len(sequence) < args.frames:
                sequence.append(crop)
                if len(sequence) == args.frames:
                    recording = False
                    capture_message = "READY TO SAVE: press Enter"
                    capture_message_until = time.monotonic() + 10
            j_count = len(list((args.output_dir / "J").glob("*.npy")))
            z_count = len(list((args.output_dir / "Z").glob("*.npy")))
            if recording:
                status = f"Recording {label} | Frames: {len(sequence)}/{args.frames}"
            elif len(sequence) == args.frames:
                status = f"Ready {label} | Frames: {len(sequence)}/{args.frames}"
            else:
                status = f"Label: {label} | J saved: {j_count} | Z saved: {z_count}"
            cv2.putText(display, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, "J/Z label | Space start | Enter save | Backspace cancel | Esc quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            if time.monotonic() < capture_message_until:
                cv2.putText(display, capture_message, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Collect motion sign", display)
            if crop is not None:
                cv2.imshow("Motion crop", crop)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("j"), ord("J")) and not recording and not sequence:
                label = "J"
            if key in (ord("z"), ord("Z")) and not recording and not sequence:
                label = "Z"
            if key == 32 and not recording and not sequence:
                if crop is not None:
                    sequence = [crop]
                    recording = True
                else:
                    capture_message = "NO HAND DETECTED: cannot start"
                    capture_message_until = time.monotonic() + 2
            if key == 13 and len(sequence) == args.frames:
                label_dir = args.output_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)
                index = len(list(label_dir.glob("*.npy"))) + 1
                np.save(label_dir / f"{label}_{index:05d}.npy", np.stack(sequence))
                capture_message = f"SAVED {label}_{index:05d}.npy"
                capture_message_until = time.monotonic() + 2
                sequence = []
            elif key == 13:
                capture_message = f"NOT READY: {len(sequence)}/{args.frames} frames"
                capture_message_until = time.monotonic() + 2
            if key in (8, 127):
                sequence = []
                recording = False
                capture_message = "CANCELLED"
                capture_message_until = time.monotonic() + 2
    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
