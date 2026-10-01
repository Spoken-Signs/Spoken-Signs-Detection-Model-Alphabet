import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import RunningMode
import torch
from torch import nn
from torchvision import models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time ASL fingerspelling.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).with_name("alphabet_model.pt"),
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--stable-ratio", type=float, default=0.6)
    parser.add_argument("--padding", type=float, default=0.35)
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=Path(__file__).with_name("hand_landmarker.task"),
    )
    return parser.parse_args()


def load_model(model_path):
    print(f"Loading classifier: {model_path}", flush=True)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Classifier loaded on {device}", flush=True)
    return model, classes, device


def square_hand_crop(frame, hand_landmarks, padding):
    height, width = frame.shape[:2]
    x_values = [landmark.x for landmark in hand_landmarks]
    y_values = [landmark.y for landmark in hand_landmarks]
    left = max(0, int(min(x_values) * width))
    right = min(width, int(max(x_values) * width))
    top = max(0, int(min(y_values) * height))
    bottom = min(height, int(max(y_values) * height))

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    side = max(right - left, bottom - top)
    side = int(side * (1.0 + padding))
    crop_left = max(0, center_x - side // 2)
    crop_top = max(0, center_y - side // 2)
    crop_right = min(width, crop_left + side)
    crop_bottom = min(height, crop_top + side)
    crop_left = max(0, crop_right - side)
    crop_top = max(0, crop_bottom - side)
    return frame[crop_top:crop_bottom, crop_left:crop_right], (
        crop_left,
        crop_top,
        crop_right,
        crop_bottom,
    )


def open_camera(camera_index):
    backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation")]
    for backend, backend_name in backends:
        camera = cv2.VideoCapture(camera_index, backend)
        if camera.isOpened():
            for _ in range(5):
                camera.grab()
            return camera, backend_name
        camera.release()
    raise RuntimeError(f"Could not open camera {camera_index} with DirectShow or Media Foundation")


def main():
    args = parse_args()
    model, classes, device = load_model(args.model)
    image_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    camera, camera_backend = open_camera(args.camera)
    print(f"Camera {args.camera} opened with {camera_backend}", flush=True)

    if not args.hand_model.is_file():
        raise FileNotFoundError(
            f"Hand model not found: {args.hand_model}. "
            "Download hand_landmarker.task and place it beside this script."
        )
    hand_tracker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.hand_model)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )
    print("Hand tracker ready. The camera window is now active; press Esc in that window to quit.", flush=True)
    recent_predictions = deque(maxlen=args.history)
    text = ""
    last_committed = None
    armed = True
    frame_timestamp_ms = 0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                success, frame = camera.read()
            if not success:
                raise RuntimeError(
                    f"Could not read a frame from camera {args.camera} using {camera_backend}"
                )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            tracking_result = hand_tracker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33
            display_frame = cv2.flip(frame, 1) if args.mirror else frame.copy()
            label = "No hand"
            confidence = 0.0
            crop_box = None

            if tracking_result.hand_landmarks:
                handedness = tracking_result.handedness[0][0].category_name
                hand_crop, crop_box = square_hand_crop(
                    frame,
                    tracking_result.hand_landmarks[0],
                    args.padding,
                )
                if handedness == "Left":
                    hand_crop = cv2.flip(hand_crop, 1)
                crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                image = image_transform(crop_rgb).unsqueeze(0).to(device)
                with torch.inference_mode():
                    probabilities = model(image).softmax(dim=1)[0]
                confidence_tensor, prediction_index = probabilities.max(dim=0)
                label = classes[prediction_index.item()]
                confidence = confidence_tensor.item()

                if confidence >= args.confidence:
                    recent_predictions.append(label)
                else:
                    recent_predictions.clear()

                if len(recent_predictions) == args.history:
                    stable_label, stable_count = Counter(recent_predictions).most_common(1)[0]
                    is_stable = stable_count >= max(2, round(args.history * args.stable_ratio))
                    if is_stable and armed and stable_label != "nothing":
                        if stable_label == "del":
                            text = text[:-1]
                        elif stable_label == "space":
                            text += " "
                        else:
                            text += stable_label
                        last_committed = stable_label
                        armed = False
                    if stable_label == "nothing" or stable_label != last_committed:
                        armed = True

            if crop_box is not None:
                left, top, right, bottom = crop_box
                if args.mirror:
                    frame_width = display_frame.shape[1]
                    left, right = frame_width - right, frame_width - left
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 255), 2)

            cv2.putText(
                display_frame,
                f"Sign: {label} ({confidence:.1%})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Text: {text}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "Hold sign steady; neutral between repeats; Esc quits; Backspace clears",
                (20, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("ASL fingerspelling", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (8, 127):
                text = ""
                last_committed = None
                armed = True
    finally:
        hand_tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
