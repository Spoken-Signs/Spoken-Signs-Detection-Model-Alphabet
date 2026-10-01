import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Predict an ASL alphabet class from one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).with_name("alphabet_model.pt"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    image = image_transform(Image.open(args.image).convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        probabilities = model(image).softmax(dim=1)[0]
    confidence, prediction_index = probabilities.max(dim=0)
    print(f"Prediction: {classes[prediction_index.item()]}")
    print(f"Confidence: {confidence.item():.2%}")


if __name__ == "__main__":
    main()