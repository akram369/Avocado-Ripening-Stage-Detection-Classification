"""
dataset.py - Dataset loading, stratified splitting, preprocessing, and augmentation pipeline.
Handles both Citrus (Orange) and Avocado classification datasets seamlessly.
"""

import os
import glob
from typing import Tuple, List, Dict, Optional
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def clean_load_image(image_path: str) -> Image.Image:
    """
    Loads an image from disk and safely converts to 3-channel RGB.
    If the image has an alpha channel (RGBA), it replaces transparent pixels
    with a neutral white background instead of black artifacts.
    """
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img_rgba = img.convert("RGBA")
            background = Image.new("RGBA", img_rgba.size, (255, 255, 255))
            blended = Image.alpha_composite(background, img_rgba)
            return blended.convert("RGB")
        return img.convert("RGB")


class ProduceDataset(Dataset):
    """
    Custom PyTorch Dataset for produce ripening / defect classification.
    Supports on-the-fly PIL transformations and caching if needed.
    """
    def __init__(self, file_paths: List[str], labels: List[int], transform: Optional[transforms.Compose] = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.file_paths[idx]
        label = self.labels[idx]
        image = clean_load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_default_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and validation/test transforms.
    Train transforms include robust data augmentations:
    - Random Horizontal & Vertical Flips
    - Random Rotation (-20 to +20 degrees)
    - Color Jitter (Brightness, Contrast, Saturation) to handle specular reflections and lighting shifts
    - Normalization with ImageNet standard mean and std
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


def scan_class_folders(directory: str) -> Tuple[List[str], List[str], List[int]]:
    """
    Scans a directory containing subfolders as class names.
    Returns:
        classes (sorted class names)
        filepaths (list of absolute image paths)
        labels (list of integer label indices)
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    classes = sorted([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and not d.startswith(".")
    ])
    
    filepaths = []
    labels = []

    for idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(directory, cls_name)
        for fname in os.listdir(cls_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                filepaths.append(os.path.join(cls_dir, fname))
                labels.append(idx)

    return classes, filepaths, labels


def prepare_dataset_splits(
    base_dir: str = ".",
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict:
    """
    Prepares stratified train, val, and test splits.
    Checks for existing train/ and test/ directories, or raw directory.
    """
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    if not os.path.exists(train_dir):
        # Fallback to data/raw
        train_dir = os.path.join(base_dir, "data", "raw", "train")
        test_dir = os.path.join(base_dir, "data", "raw", "test")

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Existing train and test directories
        train_classes, train_paths, train_labels = scan_class_folders(train_dir)
        test_classes, test_paths, test_labels = scan_class_folders(test_dir)
        
        assert train_classes == test_classes, f"Classes mismatch between train {train_classes} and test {test_classes}"
        classes = train_classes

        # Stratified train/val split from train
        tr_paths, val_paths, tr_labels, val_labels = train_test_split(
            train_paths,
            train_labels,
            test_size=val_ratio,
            stratify=train_labels,
            random_state=random_seed
        )
        te_paths, te_labels = test_paths, test_labels

    else:
        # Search all data in data/ or base_dir
        scan_dir = os.path.join(base_dir, "data", "raw") if os.path.exists(os.path.join(base_dir, "data", "raw")) else base_dir
        classes, all_paths, all_labels = scan_class_folders(scan_dir)

        # 70% Train, 15% Val, 15% Test Stratified Split
        tr_paths, temp_paths, tr_labels, temp_labels = train_test_split(
            all_paths, all_labels, test_size=0.30, stratify=all_labels, random_state=random_seed
        )
        val_paths, te_paths, val_labels, te_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=random_seed
        )

    # Compute balanced class weights for training
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(tr_labels),
        y=tr_labels
    )
    class_weights = torch.tensor(class_weights_arr, dtype=torch.float32)

    return {
        "classes": classes,
        "num_classes": len(classes),
        "train": (tr_paths, tr_labels),
        "val": (val_paths, val_labels),
        "test": (te_paths, te_labels),
        "class_weights": class_weights,
    }


def get_dataloaders(
    splits: Dict,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds PyTorch DataLoaders for train, val, and test partitions.
    """
    train_transform, eval_transform = get_default_transforms(img_size)

    train_ds = ProduceDataset(splits["train"][0], splits["train"][1], transform=train_transform)
    val_ds = ProduceDataset(splits["val"][0], splits["val"][1], transform=eval_transform)
    test_ds = ProduceDataset(splits["test"][0], splits["test"][1], transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
