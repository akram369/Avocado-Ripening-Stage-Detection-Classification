
"""
main_demo.py - Single-run inference and live demonstration script for Greenpod Labs Assignment 2.
Evaluates any test image with both Approach 1 (Classical ML) and Approach 2 (MobileNetV2 Transfer Learning),
measuring inference latency, class probabilities, and automated produce sorting decisions.
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
from skimage.feature import local_binary_pattern

from src.dataset import clean_load_image, get_default_transforms
from src.classical_model import ClassicalClassifierPipeline, extract_features_from_image
from src.dl_model import ProduceClassifierCNN


SORTING_ACTIONS = {
    "fresh": "GRADE A (FRESH): Approved for primary retail distribution & long-distance export.",
    "grenning": "STAGE: BREAKING / CITRUS GREENING: Divert to controlled ripening room / shelf-life formulation application.",
    "canker": "DEFECT (CANKER LESIONS): Quarantine / reject for fresh retail; divert to industrial extraction/processing.",
    "blackspot": "DEFECT (BLACKSPOT DECAY): Surface decay detected; sort out to prevent fungal contamination.",
    "underripe": "UNDER-RIPE / HARD: Store in climate-controlled ripening chambers.",
    "breaking": "BREAKING / FIRM: Near-ready; schedule for immediate regional retail.",
    "ripe": "RIPE / READY-TO-EAT: Optimal eating stage; prioritize for immediate point-of-sale display.",
    "overripe": "OVERRIPE / DECAY: Divert to bio-waste or industrial processing."
}


def load_trained_models(
    svm_path: str = "models/classical_svm.joblib",
    dl_path: str = "models/mobilenetv2_best.pth",
    device: Optional[torch.device] = None
):
    """Loads both the Classical SVM model and the Deep Learning MobileNetV2 model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(svm_path):
        raise FileNotFoundError(f"Classical model file not found at '{svm_path}'. Please run train_and_evaluate.py first.")
    if not os.path.exists(dl_path):
        raise FileNotFoundError(f"Deep learning checkpoint not found at '{dl_path}'. Please run train_and_evaluate.py first.")

    # 1. Classical SVM
    classical_pipeline = ClassicalClassifierPipeline.load(svm_path)
    classes = classical_pipeline.classes_

    # 2. Deep Learning MobileNetV2
    checkpoint = torch.load(dl_path, map_location=device)
    dl_model = ProduceClassifierCNN(num_classes=len(classes), architecture="mobilenet_v2", pretrained=False)
    dl_model.load_state_dict(checkpoint["model_state_dict"])
    dl_model.to(device)
    dl_model.eval()

    return classical_pipeline, dl_model, classes, device


def run_avocado_demo(
    image_path: str,
    classical_model: ClassicalClassifierPipeline,
    dl_model: ProduceClassifierCNN,
    class_names: list,
    device: torch.device,
    save_viz_path: Optional[str] = "demo_prediction_result.png"
):
    """
    Executes single-image dual-model inference and outputs comparative benchmark.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("\n" + "=" * 75)
    print(f"AUTOMATED PRODUCE INSPECTION: INFERENCE DEMO")
    print(f"Target Image: {image_path}")
    print("=" * 75)

    # Load and clean image
    pil_img = clean_load_image(image_path)
    img_np = np.array(pil_img)

    # -------------------------------------------------------------
    # 1. Approach 1: Classical CV Feature Extraction + SVM
    # -------------------------------------------------------------
    t0_feat = time.perf_counter()
    features = extract_features_from_image(img_np)
    t1_feat = time.perf_counter()

    t0_class = time.perf_counter()
    classical_pred_idx = classical_model.predict(features.reshape(1, -1))[0]
    classical_probs = classical_model.predict_proba(features.reshape(1, -1))[0]
    t1_class = time.perf_counter()

    classical_pred_name = class_names[classical_pred_idx]
    classical_conf = classical_probs[classical_pred_idx] * 100
    feat_time_ms = (t1_feat - t0_feat) * 1000
    class_time_ms = (t1_class - t0_class) * 1000
    classical_total_ms = feat_time_ms + class_time_ms

    # -------------------------------------------------------------
    # 2. Approach 2: Deep Learning MobileNetV2
    # -------------------------------------------------------------
    _, eval_transform = get_default_transforms(img_size=224)
    tensor = eval_transform(pil_img).unsqueeze(0).to(device)

    t0_dl = time.perf_counter()
    with torch.no_grad():
        dl_outputs = dl_model(tensor)
        dl_probs = torch.softmax(dl_outputs, dim=1).squeeze(0).cpu().numpy()
    t1_dl = time.perf_counter()

    dl_pred_idx = int(np.argmax(dl_probs))
    dl_pred_name = class_names[dl_pred_idx]
    dl_conf = dl_probs[dl_pred_idx] * 100
    dl_time_ms = (t1_dl - t0_dl) * 1000

    # -------------------------------------------------------------
    # Print Comparison Report
    # -------------------------------------------------------------
    print("\n[PREDICTION RESULTS]")
    print(f"Approach 1 (Classical CV + SVM):    {classical_pred_name.upper():<12} (Confidence: {classical_conf:.1f}%) | Latency: {classical_total_ms:.1f} ms")
    print(f"Approach 2 (Deep Learning MobileNet): {dl_pred_name.upper():<12} (Confidence: {dl_conf:.1f}%) | Latency: {dl_time_ms:.1f} ms")

    print("\n[CLASS PROBABILITY DISTRIBUTION]")
    print(f"{'Class':<15} {'Classical SVM':<16} {'MobileNetV2':<16}")
    print("-" * 47)
    for i, c in enumerate(class_names):
        print(f"{c:<15} {classical_probs[i]*100:6.2f}%          {dl_probs[i]*100:6.2f}%")

    sorting_action = SORTING_ACTIONS.get(dl_pred_name, "Divert to secondary grading.")
    print("\n[AUTOMATED CONVEYOR SORTING ACTION]")
    print(f">> {sorting_action}")
    print("=" * 75)

    # -------------------------------------------------------------
    # Visual Explanation Plot
    # -------------------------------------------------------------
    if save_viz_path:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        
        # 1. Original Image
        axes[0].imshow(img_np)
        axes[0].set_title(f"Input Sample\nTrue Stage/Defect", fontweight='bold')
        axes[0].axis('off')

        # 2. CIELAB a* Map (Green-Red Transitions)
        img_resized = cv2.resize(img_np, (150, 150))
        lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
        im_a = axes[1].imshow(lab[:, :, 1], cmap='PiYG')
        axes[1].set_title("CIELAB a* Chromaticity\n(Green-to-Red Shift)", fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im_a, ax=axes[1], fraction=0.046, pad=0.04)

        # 3. LBP Texture Map
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        im_lbp = axes[2].imshow(lbp, cmap='inferno')
        axes[2].set_title("LBP Texture Map\n(Surface Roughness / Pitting)", fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im_lbp, ax=axes[2], fraction=0.046, pad=0.04)

        # 4. Probability Comparison Bar Chart
        x = np.arange(len(class_names))
        width = 0.35
        axes[3].bar(x - width/2, classical_probs * 100, width, label='SVM (Classical)', color='#1f77b4')
        axes[3].bar(x + width/2, dl_probs * 100, width, label='MobileNetV2', color='#2ca02c')
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(class_names, rotation=25, ha='right', fontweight='bold')
        axes[3].set_ylabel('Probability (%)', fontweight='bold')
        axes[3].set_title("Model Confidence Comparison", fontweight='bold')
        axes[3].set_ylim(0, 105)
        axes[3].legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(save_viz_path, dpi=200)
        plt.close()
        print(f"\nVisual inference explanation saved to: {save_viz_path}")

    return {
        "classical_pred": classical_pred_name,
        "classical_probs": classical_probs,
        "classical_latency_ms": classical_total_ms,
        "dl_pred": dl_pred_name,
        "dl_probs": dl_probs,
        "dl_latency_ms": dl_time_ms,
    }


def find_default_sample():
    """Finds a test sample image from test directory."""
    for root, dirs, files in os.walk("test"):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(root, f)
    # Fallback to train
    for root, dirs, files in os.walk("train"):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(root, f)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avocado / Produce Ripening Stage & Defect Detection Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--output", type=str, default="demo_prediction_result.png", help="Path to save visual output")
    args = parser.parse_args()

    image_to_test = args.image
    if image_to_test is None:
        image_to_test = find_default_sample()
        if image_to_test is None:
            print("Error: No images found in test/ or train/ directories.")
            sys.exit(1)
        print(f"No --image argument provided. Using sample from test set: {image_to_test}")

    classical_model, dl_model, class_names, device = load_trained_models()
    run_avocado_demo(
        image_path=image_to_test,
        classical_model=classical_model,
        dl_model=dl_model,
        class_names=class_names,
        device=device,
        save_viz_path=args.output
    )
