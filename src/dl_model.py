"""
dl_model.py - Approach 2: Deep Learning Transfer Learning (MobileNetV2 / ResNet).
Implements a lightweight edge-optimized Convolutional Neural Network with custom classification head,
class-weighted CrossEntropyLoss, early stopping, and metric tracking.
"""

import os
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from PIL import Image

from src.dataset import clean_load_image, get_default_transforms


class ProduceClassifierCNN(nn.Module):
    """
    Transfer learning wrapper for MobileNetV2 (or ResNet18).
    Designed specifically for edge processing in sorting conveyor lines.
    """
    def __init__(self, num_classes: int = 4, architecture: str = "mobilenet_v2", pretrained: bool = True):
        super().__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes

        if self.architecture == "mobilenet_v2":
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            
            # Custom edge classification head with regularizations
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes)
            )

        elif self.architecture == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes)
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}. Choose 'mobilenet_v2' or 'resnet18'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """Freezes all feature extraction layers, training only the classification head."""
        if self.architecture == "mobilenet_v2":
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        elif self.architecture == "resnet18":
            for name, param in self.backbone.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    def unfreeze_top_layers(self, num_layers: int = 4):
        """Unfreezes top inverted residual blocks for fine-tuning."""
        if self.architecture == "mobilenet_v2":
            for layer in self.backbone.features[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True


def train_dl_model(
    model: ProduceClassifierCNN,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    class_weights: Optional[torch.Tensor] = None,
    save_path: str = "models/mobilenetv2_best.pth",
    patience: int = 5
) -> Dict:
    """
    Trains the CNN model with validation monitoring, early stopping, and learning rate decay.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.to(device)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "epoch_time": []
    }

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting Training on {device} for {epochs} epochs...")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(elapsed)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"Time: {elapsed:.1f}s")

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "architecture": model.architecture,
                "num_classes": model.num_classes
            }, save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs (Patience {patience}).")
                break

    # Reload best checkpoint weights
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Restored best weights from epoch {checkpoint['epoch']} with Val Loss: {checkpoint['val_loss']:.4f}")

    return history


def evaluate_dl_model(
    model: ProduceClassifierCNN,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict:
    """
    Evaluates the model on test set, computing accuracy, F1-scores, confusion matrix, and latency.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    latencies = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            t_start = time.perf_counter()
            outputs = model(images)
            t_end = time.perf_counter()

            latencies.append((t_end - t_start) / images.size(0))
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    avg_latency_ms = np.mean(latencies) * 1000

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "probabilities": all_probs,
        "predictions": all_preds,
        "targets": all_targets,
        "avg_latency_ms": avg_latency_ms
    }


def predict_single_image(
    model: ProduceClassifierCNN,
    image_path: str,
    device: torch.device,
    class_names: List[str],
    img_size: int = 224
) -> Tuple[str, float, np.ndarray, float]:
    """
    Runs single-image inference and returns (predicted_class, confidence, probabilities, latency_ms).
    """
    model.eval()
    _, eval_transform = get_default_transforms(img_size)
    img = clean_load_image(image_path)
    tensor = eval_transform(img).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
    t1 = time.perf_counter()

    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    latency_ms = (t1 - t0) * 1000

    return class_names[pred_idx], conf, probs, latency_ms
