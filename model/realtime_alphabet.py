import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import torch
from torch import nn
from torchvision import models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Recognize ASL alphabet signs from a webcam.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).with_name("alphabet_model.pt"),
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--roi-scale", type=float, default=0.8)
    return parser.parse_args()


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, classes, device


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

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    recent_predictions = deque(maxlen=7)
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera")

            height, width = frame.shape[:2]
            side = int(min(height, width) * args.roi_scale)
            left = (width - side) // 2
            top = (height - side) // 2
            hand_region = frame[top : top + side, left : left + side]

            rgb_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
            image = image_transform(rgb_region).unsqueeze(0).to(device)
            with torch.inference_mode():
                probabilities = model(image).softmax(dim=1)[0]
            confidence, prediction_index = probabilities.max(dim=0)
            label = classes[prediction_index.item()]
            recent_predictions.append(label)
            stable_label = Counter(recent_predictions).most_common(1)[0][0]

            display_frame = cv2.flip(frame, 1) if args.mirror else frame.copy()
            display_left = width - left - side if args.mirror else left
            display_right = display_left + side
            cv2.rectangle(
                display_frame,
                (display_left, top),
                (display_right, top + side),
                (0, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"{stable_label} ({confidence.item():.1%})",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "Press q to quit",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("ASL alphabet", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()