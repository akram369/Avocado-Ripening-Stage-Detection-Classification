"""
train_and_evaluate.py - End-to-End Orchestrator for Greenpod Labs Assignment 2.
Performs:
1. Dataset Analysis & Bias Verification
2. Figure generation (Distribution, Resolutions, CIELAB a* / HSV, LBP Texture Maps)
3. Approach 1: Classical Feature Extraction + SVM / Random Forest Training & Evaluation
4. Approach 2: Deep Learning MobileNetV2 Transfer Learning Training & Evaluation
5. Metrics Comparative Benchmarking (Accuracy, Precision, Recall, F1, Latency)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

import torch
from skimage.feature import local_binary_pattern

from src.dataset import prepare_dataset_splits, get_dataloaders, clean_load_image
from src.classical_model import (
    extract_features_batch,
    ClassicalClassifierPipeline
)
from src.dl_model import (
    ProduceClassifierCNN,
    train_dl_model,
    evaluate_dl_model
)

# Set high-quality plot aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11


def generate_eda_figures(splits, figures_dir="figures"):
    """Generates and saves exploratory data analysis plots."""
    os.makedirs(figures_dir, exist_ok=True)
    classes = splits["classes"]

    tr_paths, tr_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    te_paths, te_labels = splits["test"]

    # 1. Class Distribution Plot
    train_counts = [sum(1 for l in tr_labels + val_labels if l == i) for i in range(len(classes))]
    test_counts = [sum(1 for l in te_labels if l == i) for i in range(len(classes))]

    df_dist = pd.DataFrame({
        'Class': classes * 2,
        'Count': train_counts + test_counts,
        'Split': ['Train + Val'] * len(classes) + ['Test'] * len(classes)
    })

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df_dist, x='Class', y='Count', hue='Split', palette=['#2b5c8f', '#e28743'])
    plt.title("Produce Dataset Class Distribution Across Splits", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Category / Ripening & Defect Class", fontweight='bold')
    plt.ylabel("Number of Images", fontweight='bold')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height + 4),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "01_class_distribution.png"), dpi=200)
    plt.close()
    print("Saved 01_class_distribution.png")

    # 2. Resolution & Aspect Ratio Bias Plot
    records = []
    for split_name, (paths, labels) in [("Train", (tr_paths, tr_labels)), ("Test", (te_paths, te_labels))]:
        for p, l in zip(paths, labels):
            with Image.open(p) as img:
                w, h = img.size
                records.append({
                    "split": split_name,
                    "class": classes[l],
                    "width": w,
                    "height": h,
                    "format": img.format
                })
    df_res = pd.DataFrame(records)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_res, x='width', y='height', hue='class', style='format', alpha=0.7, s=70)
    plt.title("Original Image Resolutions by Class", fontweight='bold')
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_res, x='class', y='width', palette='Set2')
    plt.title("Resolution Shift Artifact (Fresh vs Diseases)", fontweight='bold')
    plt.xlabel("Class")
    plt.ylabel("Image Width (px)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "02_resolution_bias_analysis.png"), dpi=200)
    plt.close()
    print("Saved 02_resolution_bias_analysis.png")

    # 3. CIELAB a* & HSV Hue Distribution Analysis
    sample_images_per_class = {}
    for p, l in zip(tr_paths, tr_labels):
        if classes[l] not in sample_images_per_class:
            sample_images_per_class[classes[l]] = p
        if len(sample_images_per_class) == len(classes):
            break

    fig, axes = plt.subplots(len(classes), 3, figsize=(12, 3.2 * len(classes)))
    for idx, (cls_name, img_path) in enumerate(sample_images_per_class.items()):
        img_pil = clean_load_image(img_path)
        img_np = np.array(img_pil.resize((150, 150)))
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # Original
        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f"Class: {cls_name}", fontweight='bold')
        axes[idx, 0].axis('off')

        # CIELAB a* channel (Green-to-Red)
        a_channel = lab[:, :, 1]
        im1 = axes[idx, 1].imshow(a_channel, cmap='PiYG')
        axes[idx, 1].set_title(f"{cls_name} - CIELAB a* (Green/Red)", fontsize=10)
        axes[idx, 1].axis('off')

        # HSV Hue channel
        h_channel = hsv[:, :, 0]
        axes[idx, 2].imshow(h_channel, cmap='hsv')
        axes[idx, 2].set_title(f"{cls_name} - HSV Hue Map", fontsize=10)
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "03_color_space_lab_hsv.png"), dpi=200)
    plt.close()
    print("Saved 03_color_space_lab_hsv.png")

    # 4. LBP Texture Analysis Maps
    fig, axes = plt.subplots(len(classes), 3, figsize=(11, 3.0 * len(classes)))
    for idx, (cls_name, img_path) in enumerate(sample_images_per_class.items()):
        img_pil = clean_load_image(img_path)
        img_np = np.array(img_pil.resize((150, 150)))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        lbp_fine = local_binary_pattern(gray, P=8, R=1, method='uniform')

        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f"{cls_name} (RGB)", fontweight='bold')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(gray, cmap='gray')
        axes[idx, 1].set_title(f"Grayscale", fontsize=10)
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(lbp_fine, cmap='inferno')
        axes[idx, 2].set_title(f"Uniform LBP Texture Map", fontsize=10)
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "04_lbp_texture_analysis.png"), dpi=200)
    plt.close()
    print("Saved 04_lbp_texture_analysis.png")


def plot_confusion_matrix(cm, classes, title, save_path):
    """Plots and saves normalized confusion matrix."""
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, cbar=True, annot_kws={"size": 13, "weight": "bold"})
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Ground Truth Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_full_pipeline():
    print("=" * 70)
    print("GREENPOD LABS ASSIGNMENT 2: COMPUTER VISION CLASSIFICATION WORKFLOW")
    print("=" * 70)

    # 1. Dataset Preparation
    print("\n[Step 1/5] Loading Dataset & Preparing Stratified Partitions...")
    splits = prepare_dataset_splits(base_dir=".", val_ratio=0.15, random_seed=42)
    classes = splits["classes"]
    print(f"Identified {splits['num_classes']} Classes: {classes}")

    tr_paths, tr_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    te_paths, te_labels = splits["test"]

    print(f"Dataset Partitions: Train={len(tr_paths)}, Val={len(val_paths)}, Test={len(te_paths)}")

    # 2. EDA Figure Generation
    print("\n[Step 2/5] Generating Exploratory Data Analysis & Bias Plots...")
    generate_eda_figures(splits, figures_dir="figures")

    # 3. Approach 1: Classical Feature Extraction + Machine Learning
    print("\n[Step 3/5] Approach 1: Classical CV Feature Extraction (CIELAB + HSV + LBP + GLCM)...")
    t0_feat = time.time()
    print(f"  Extracting features for {len(tr_paths)} Train samples...")
    X_train = extract_features_batch(tr_paths)
    y_train = np.array(tr_labels)

    print(f"  Extracting features for {len(val_paths)} Val samples...")
    X_val = extract_features_batch(val_paths)
    y_val = np.array(val_labels)

    print(f"  Extracting features for {len(te_paths)} Test samples...")
    X_test = extract_features_batch(te_paths)
    y_test = np.array(te_labels)
    feat_time = time.time() - t0_feat
    print(f"Feature Extraction complete in {feat_time:.1f}s. Feature vector shape: {X_train.shape}")

    # Combine Train + Val for final classical model training
    X_tr_full = np.vstack([X_train, X_val])
    y_tr_full = np.concatenate([y_train, y_val])

    # Train SVM
    print("  Training Approach 1A: Support Vector Machine (RBF Kernel)...")
    svm_pipeline = ClassicalClassifierPipeline(model_type="svm")
    t0_svm = time.time()
    svm_pipeline.fit(X_tr_full, y_tr_full, class_names=classes)
    t_svm_train = time.time() - t0_svm
    svm_pipeline.save("models/classical_svm.joblib")

    # Benchmark SVM test latency
    t0_inf = time.perf_counter()
    svm_eval = svm_pipeline.evaluate(X_test, y_test)
    svm_latency_ms = ((time.perf_counter() - t0_inf) / len(X_test)) * 1000
    svm_eval["avg_latency_ms"] = svm_latency_ms
    svm_eval["train_time_s"] = t_svm_train

    print(f"  --> SVM Test Accuracy: {svm_eval['accuracy']*100:.2f}% | Macro F1: {svm_eval['macro_f1']:.4f} | Latency: {svm_latency_ms:.2f}ms/img")
    plot_confusion_matrix(svm_eval["confusion_matrix"], classes, "Approach 1 (Classical SVM) Confusion Matrix", "figures/05_classical_svm_confusion_matrix.png")

    # Train Random Forest
    print("  Training Approach 1B: Random Forest Classifier (200 Trees)...")
    rf_pipeline = ClassicalClassifierPipeline(model_type="rf")
    t0_rf = time.time()
    rf_pipeline.fit(X_tr_full, y_tr_full, class_names=classes)
    t_rf_train = time.time() - t0_rf
    rf_pipeline.save("models/classical_rf.joblib")

    t0_inf_rf = time.perf_counter()
    rf_eval = rf_pipeline.evaluate(X_test, y_test)
    rf_latency_ms = ((time.perf_counter() - t0_inf_rf) / len(X_test)) * 1000
    rf_eval["avg_latency_ms"] = rf_latency_ms
    rf_eval["train_time_s"] = t_rf_train

    print(f"  --> Random Forest Test Accuracy: {rf_eval['accuracy']*100:.2f}% | Macro F1: {rf_eval['macro_f1']:.4f} | Latency: {rf_latency_ms:.2f}ms/img")
    plot_confusion_matrix(rf_eval["confusion_matrix"], classes, "Approach 1 (Random Forest) Confusion Matrix", "figures/06_classical_rf_confusion_matrix.png")

    # 4. Approach 2: Deep Learning (MobileNetV2 Transfer Learning)
    print("\n[Step 4/5] Approach 2: Deep Learning Transfer Learning (MobileNetV2)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(splits, batch_size=32, img_size=224, num_workers=0)
    dl_model = ProduceClassifierCNN(num_classes=len(classes), architecture="mobilenet_v2", pretrained=True)
    
    # Train classification head with frozen backbone first
    dl_model.freeze_backbone()
    dl_history = train_dl_model(
        model=dl_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=12,
        lr=1e-3,
        class_weights=splits["class_weights"],
        save_path="models/mobilenetv2_best.pth",
        patience=4
    )

    # Plot DL Training Curves
    plt.figure(figsize=(11, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(dl_history["train_loss"], label="Train Loss", color='#2b5c8f', linewidth=2)
    plt.plot(dl_history["val_loss"], label="Val Loss", color='#e28743', linewidth=2, linestyle='--')
    plt.title("MobileNetV2 Training & Validation Loss", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([a * 100 for a in dl_history["train_acc"]], label="Train Acc", color='#2b5c8f', linewidth=2)
    plt.plot([a * 100 for a in dl_history["val_acc"]], label="Val Acc", color='#e28743', linewidth=2, linestyle='--')
    plt.title("MobileNetV2 Training & Validation Accuracy (%)", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/07_dl_training_curves.png", dpi=200)
    plt.close()
    print("Saved 07_dl_training_curves.png")

    # Evaluate DL Model on Test Set
    dl_eval = evaluate_dl_model(dl_model, test_loader, device=device, class_names=classes)
    print(f"  --> MobileNetV2 Test Accuracy: {dl_eval['accuracy']*100:.2f}% | Macro F1: {dl_eval['macro_f1']:.4f} | Latency: {dl_eval['avg_latency_ms']:.2f}ms/img")
    plot_confusion_matrix(dl_eval["confusion_matrix"], classes, "Approach 2 (MobileNetV2 Transfer Learning) Confusion Matrix", "figures/08_dl_mobilenet_confusion_matrix.png")

    # 5. Model Comparison & Summary
    print("\n[Step 5/5] Generating Model Comparison & Benchmark Summary...")
    comparison_data = [
        {
            "Model Approach": "Approach 1A: Classical SVM (CIELAB/LBP)",
            "Accuracy (%)": round(svm_eval["accuracy"] * 100, 2),
            "Macro F1": round(svm_eval["macro_f1"], 4),
            "Weighted F1": round(svm_eval["weighted_f1"], 4),
            "Inference Latency (ms)": round(svm_latency_ms, 2),
            "Training Time (s)": round(svm_eval["train_time_s"] + feat_time, 2)
        },
        {
            "Model Approach": "Approach 1B: Classical Random Forest",
            "Accuracy (%)": round(rf_eval["accuracy"] * 100, 2),
            "Macro F1": round(rf_eval["macro_f1"], 4),
            "Weighted F1": round(rf_eval["weighted_f1"], 4),
            "Inference Latency (ms)": round(rf_latency_ms, 2),
            "Training Time (s)": round(rf_eval["train_time_s"] + feat_time, 2)
        },
        {
            "Model Approach": "Approach 2: Deep Learning (MobileNetV2)",
            "Accuracy (%)": round(dl_eval["accuracy"] * 100, 2),
            "Macro F1": round(dl_eval["macro_f1"], 4),
            "Weighted F1": round(dl_eval["weighted_f1"], 4),
            "Inference Latency (ms)": round(dl_eval["avg_latency_ms"], 2),
            "Training Time (s)": round(sum(dl_history["epoch_time"]), 2)
        }
    ]

    df_comp = pd.DataFrame(comparison_data)
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    print(df_comp.to_string(index=False))

    # Comparative Bar Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_comp))
    width = 0.25

    rects1 = ax1.bar(x - width, df_comp["Accuracy (%)"], width, label="Accuracy (%)", color="#1f77b4")
    rects2 = ax1.bar(x, df_comp["Macro F1"] * 100, width, label="Macro F1 (x100)", color="#2ca02c")
    rects3 = ax1.bar(x + width, df_comp["Weighted F1"] * 100, width, label="Weighted F1 (x100)", color="#ff7f0e")

    ax1.set_ylabel("Score (%)", fontweight='bold')
    ax1.set_title("Benchmarking Comparison: Classical Feature Extraction vs MobileNetV2", fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Classical SVM", "Random Forest", "MobileNetV2 CNN"], fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 115)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("figures/09_model_comparison_benchmark.png", dpi=200)
    plt.close()
    print("Saved 09_model_comparison_benchmark.png")

    # Serialize complete benchmark metrics
    summary = {
        "classes": classes,
        "splits_count": {
            "train": len(tr_paths),
            "val": len(val_paths),
            "test": len(te_paths)
        },
        "comparison_table": comparison_data,
        "svm_classification_report": svm_eval["classification_report"],
        "rf_classification_report": rf_eval["classification_report"],
        "dl_classification_report": dl_eval["classification_report"]
    }

    os.makedirs("report", exist_ok=True)
    with open("report/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("Saved report/metrics_summary.json")

    print("\n" + "=" * 70)
    print("ALL TRAINING & EVALUATION STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
