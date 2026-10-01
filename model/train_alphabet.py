import argparse
import json
import random
from pathlib import Path

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ASL alphabet image classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parents[1]
        / "Kaggle (just alphabet) Dataset"
        / "asl_alphabet_train"
        / "asl_alphabet_train",
    )
    parser.add_argument("--camera-data-dir", type=Path, default=None)
    parser.add_argument("--camera-only", action="store_true")
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("alphabet_model.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--backbone-learning-rate", type=float, default=0.0001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class FixedClassImageFolder(datasets.ImageFolder):
    def __init__(self, root, classes, transform=None):
        self.fixed_classes = classes
        super().__init__(root, transform=transform, allow_empty=True)

    def find_classes(self, directory):
        class_to_index = {class_name: index for index, class_name in enumerate(self.fixed_classes)}
        return list(self.fixed_classes), class_to_index


def make_model(class_count):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model, weights


def split_indices(targets, validation_fraction, seed, limit_per_class=None):
    random_generator = random.Random(seed)
    indices_by_class = {}
    for index, target in enumerate(targets):
        indices_by_class.setdefault(target, []).append(index)

    train_indices = []
    validation_indices = []
    for class_indices in indices_by_class.values():
        random_generator.shuffle(class_indices)
        if limit_per_class is not None:
            class_indices = class_indices[:limit_per_class]
        validation_count = max(1, round(len(class_indices) * validation_fraction))
        validation_indices.extend(class_indices[:validation_count])
        train_indices.extend(class_indices[validation_count:])

    random_generator.shuffle(train_indices)
    random_generator.shuffle(validation_indices)
    return train_indices, validation_indices


def run_epoch(model, loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = loss_function(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    predictions = []
    labels_seen = []

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_function(outputs, labels).item() * labels.size(0)
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            labels_seen.extend(labels.cpu().tolist())

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        labels_seen,
        predictions,
        average="macro",
        zero_division=0,
    )
    accuracy = sum(
        prediction == label
        for prediction, label in zip(predictions, labels_seen)
    ) / len(labels_seen)
    return total_loss / len(labels_seen), accuracy, precision, recall, f1_score


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {args.data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, weights = make_model(class_count=29)
    if args.initial_checkpoint is not None:
        checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["classes"] != datasets.ImageFolder(args.data_dir).classes:
            raise ValueError("Initial checkpoint classes do not match the dataset classes")
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded initial checkpoint: {args.initial_checkpoint}")
    image_transforms = weights.transforms()
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    base_dataset = datasets.ImageFolder(args.data_dir)
    dataset_roots = [] if args.camera_only else [args.data_dir]
    if args.camera_data_dir is not None:
        if not args.camera_data_dir.is_dir():
            raise FileNotFoundError(f"Camera dataset folder does not exist: {args.camera_data_dir}")
        dataset_roots.append(args.camera_data_dir)
    if not dataset_roots:
        raise ValueError("--camera-only requires --camera-data-dir")
    print(f"Dataset roots: {len(dataset_roots)}")
    combined_targets = []
    for root in dataset_roots:
        combined_targets.extend(FixedClassImageFolder(root, base_dataset.classes).targets)
    train_indices, validation_indices = split_indices(
        combined_targets,
        args.validation_fraction,
        args.seed,
        args.limit_per_class,
    )

    train_dataset = ConcatDataset(
        [
            FixedClassImageFolder(root, base_dataset.classes, transform=train_transforms)
            for root in dataset_roots
        ]
    )
    validation_dataset = ConcatDataset(
        [
            FixedClassImageFolder(root, base_dataset.classes, transform=image_transforms)
            for root in dataset_roots
        ]
    )
    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(validation_dataset, validation_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model.fc = nn.Linear(model.fc.in_features, len(base_dataset.classes))
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": args.backbone_learning_rate},
            {"params": model.fc.parameters(), "lr": args.learning_rate},
        ],
        weight_decay=0.0001,
    )

    print(f"Device: {device}")
    print(f"Classes: {len(base_dataset.classes)}")
    print(f"Training images: {len(train_subset)}")
    print(f"Validation images: {len(validation_subset)}")

    best_validation_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_started = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        epoch_finished = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if epoch_started is not None:
            epoch_started.record()
        else:
            import time

            epoch_start_time = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        validation_loss, validation_accuracy, precision, recall, f1_score = evaluate(
            model, validation_loader, loss_function, device
        )
        if epoch_finished is not None:
            epoch_finished.record()
            torch.cuda.synchronize()
            epoch_time_seconds = epoch_started.elapsed_time(epoch_finished) / 1000
        else:
            epoch_time_seconds = time.perf_counter() - epoch_start_time
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f}, accuracy {train_accuracy:.2%} | "
            f"validation loss {validation_loss:.4f}, accuracy {validation_accuracy:.2%} | "
            f"precision {precision:.2%}, recall {recall:.2%}, F1 {f1_score:.2%} | "
            f"time {epoch_time_seconds:.2f}s"
        )

        if validation_accuracy >= best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            args.output.parent.mkdir(parents=True, exist_ok=True)
            crop_size = image_transforms.crop_size
            if isinstance(crop_size, int):
                image_size = [crop_size, crop_size]
            else:
                image_size = list(crop_size)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": base_dataset.classes,
                    "image_size": image_size,
                },
                args.output,
            )

    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "classes": base_dataset.classes,
                "best_validation_accuracy": best_validation_accuracy,
                "data_dir": str(args.data_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()