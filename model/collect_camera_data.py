import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import RunningMode


def parse_args():
    parser = argparse.ArgumentParser(description="Collect hand crops from a webcam for ASL fine-tuning.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("camera_data"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hand-model", type=Path, default=Path(__file__).with_name("hand_landmarker.task"))
    parser.add_argument("--padding", type=float, default=0.35)
    return parser.parse_args()


def hand_crop(frame, landmarks, padding):
    height, width = frame.shape[:2]
    x_values = [landmark.x for landmark in landmarks]
    y_values = [landmark.y for landmark in landmarks]
    left = max(0, int(min(x_values) * width))
    right = min(width, int(max(x_values) * width))
    top = max(0, int(min(y_values) * height))
    bottom = min(height, int(max(y_values) * height))
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    side = int(max(right - left, bottom - top) * (1.0 + padding))
    crop_left = max(0, center_x - side // 2)
    crop_top = max(0, center_y - side // 2)
    crop_right = min(width, crop_left + side)
    crop_bottom = min(height, crop_top + side)
    crop_left = max(0, crop_right - side)
    crop_top = max(0, crop_bottom - side)
    return frame[crop_top:crop_bottom, crop_left:crop_right]


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
    current_label = "A"
    frame_timestamp_ms = 0
    capture_message = ""
    capture_message_until = 0.0
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a camera frame")
            display = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame),
                frame_timestamp_ms,
            )
            frame_timestamp_ms += 33
            crop = None
            if result.hand_landmarks:
                crop = hand_crop(frame, result.hand_landmarks[0], args.padding)
                if result.handedness[0][0].category_name == "Left":
                    crop = cv2.flip(crop, 1)

            label_dir = args.output_dir / current_label
            captured_count = len(list(label_dir.glob("*.jpg"))) if label_dir.is_dir() else 0
            cv2.putText(display, f"Label: {current_label}  Captured: {captured_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "A-Z label | Enter capture | Backspace clear | Esc quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            if time.monotonic() < capture_message_until:
                cv2.putText(display, capture_message, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            if crop is not None:
                cv2.imshow("Crop", crop)
            cv2.imshow("Collect ASL data", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if ord("a") <= key <= ord("z"):
                current_label = chr(key).upper()
            if key == 13 and crop is not None:
                label_dir = args.output_dir / current_label
                label_dir.mkdir(parents=True, exist_ok=True)
                image_number = len(list(label_dir.glob("*.jpg"))) + 1
                output_path = label_dir / f"{current_label}_{image_number:05d}.jpg"
                temporary_path = output_path.with_suffix(".tmp.jpg")
                saved = cv2.imwrite(str(temporary_path), crop)
                if saved:
                    temporary_path.replace(output_path)
                if saved:
                    capture_message = f"CAPTURED: {output_path.name}"
                else:
                    capture_message = "CAPTURE FAILED"
                capture_message_until = time.monotonic() + 1.5
            if key in (8, 127):
                current_label = "A"
    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()