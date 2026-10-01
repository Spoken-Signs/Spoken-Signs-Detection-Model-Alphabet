import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms


class MotionDataset(Dataset):
    def __init__(self, root, transform):
        self.classes = ["J", "Z"]
        self.items = []
        for label_index, label in enumerate(self.classes):
            for path in sorted((root / label).glob("*.npy")):
                self.items.append((path, label_index))
        if not self.items:
            raise FileNotFoundError(f"No .npy sequences found under {root / 'J'} or {root / 'Z'}")
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        frames = np.load(path)
        frames = torch.stack([self.transform(frame) for frame in frames])
        return frames, label


class MotionModel(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        encoder = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.temporal = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(512, class_count)

    def forward(self, frames):
        batch, time, channels, height, width = frames.shape
        features = self.encoder(frames.reshape(batch * time, channels, height, width))
        features = features.flatten(1).reshape(batch, time, 512)
        temporal_features, _ = self.temporal(features)
        return self.classifier(temporal_features[:, -1])


def main():
    parser = argparse.ArgumentParser(description="Train a J/Z motion classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).with_name("motion_data"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("motion_model.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = MotionDataset(args.data_dir, transform)
    validation_size = max(1, round(len(dataset) * 0.2))
    train_size = len(dataset) - validation_size
    train_set, validation_set = random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)
    model = MotionModel(len(dataset.classes)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.temporal.parameters()},
        {"params": model.classifier.parameters()},
    ], lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    print(f"Device: {device} | sequences: {len(dataset)} | train: {train_size} | validation: {validation_size}")
    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)
            optimizer.zero_grad()
            loss_function(model(frames), labels).backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for frames, labels in validation_loader:
                predictions = model(frames.to(device)).argmax(1).cpu()
                correct += (predictions == labels).sum().item()
                total += len(labels)
        accuracy = correct / total
        print(f"Epoch {epoch}/{args.epochs} | validation accuracy {accuracy:.2%}")
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save({"model_state": model.state_dict(), "classes": dataset.classes}, args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
