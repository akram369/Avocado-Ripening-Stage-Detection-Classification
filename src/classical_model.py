"""
classical_model.py - Approach 1: Classical Computer Vision Feature Engineering + Machine Learning Classifiers.
Combines CIELAB & HSV Color Space Analysis, Color Moments, Local Binary Patterns (LBP),
and Gray-Level Co-occurrence Matrix (GLCM) Texture Descriptors with SVM & Random Forest.
"""

import os
import joblib
import numpy as np
import cv2
from scipy.stats import skew
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from typing import Tuple, List, Dict, Optional, Union
from PIL import Image

from src.dataset import clean_load_image


def extract_features_from_image(image_input: Union[str, np.ndarray, Image.Image], img_size: int = 128) -> np.ndarray:
    """
    Extracts a comprehensive color & texture feature vector from a single image.
    
    Feature Engineering Breakdown:
    1. CIELAB Color Space (a* channel explicitly captures green-to-red transitions; L* captures decay darkening).
       - Histograms (16 bins each for L, a, b) -> 48 features
       - Moments (mean, std, skewness for L, a, b) -> 9 features
    2. HSV Color Space (H captures pure hue angle; S captures chromatic saturation).
       - Histograms (16 bins each for H, S) -> 32 features
       - Moments (mean, std for H, S, V) -> 6 features
    3. Texture Analysis - Local Binary Patterns (LBP):
       - Fine scale (P=8, R=1, uniform) -> 10 features
       - Coarse scale (P=16, R=2, uniform) -> 18 features
    4. Texture Analysis - GLCM (Gray-Level Co-occurrence Matrix):
       - Contrast, Dissimilarity, Homogeneity, Energy, Correlation -> 5 features
       
    Total Feature Vector Dimension: 48 + 9 + 32 + 6 + 10 + 18 + 5 = 128 features.
    """
    if isinstance(image_input, str):
        pil_img = clean_load_image(image_input)
        img_rgb = np.array(pil_img)
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.shape[2] == 4:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        img_rgb = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Standardize scale
    img_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    features = []

    # -------------------------------------------------------------
    # 1. CIELAB Color Space Analysis
    # -------------------------------------------------------------
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    hist_l = cv2.calcHist([lab], [0], None, [16], [0, 256]).flatten()
    hist_a = cv2.calcHist([lab], [1], None, [16], [0, 256]).flatten()
    hist_b = cv2.calcHist([lab], [2], None, [16], [0, 256]).flatten()

    # Normalize histograms to sum to 1
    hist_l /= (hist_l.sum() + 1e-7)
    hist_a /= (hist_a.sum() + 1e-7)
    hist_b /= (hist_b.sum() + 1e-7)
    features.extend([hist_l, hist_a, hist_b])

    # Color Moments: Mean, Std, Skewness
    lab_moments = [
        np.mean(l_channel), np.std(l_channel), float(skew(l_channel.ravel())),
        np.mean(a_channel), np.std(a_channel), float(skew(a_channel.ravel())),
        np.mean(b_channel), np.std(b_channel), float(skew(b_channel.ravel())),
    ]
    features.append(np.array(lab_moments))

    # -------------------------------------------------------------
    # 2. HSV Color Space Analysis
    # -------------------------------------------------------------
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_h /= (hist_h.sum() + 1e-7)
    hist_s /= (hist_s.sum() + 1e-7)
    features.extend([hist_h, hist_s])

    hsv_moments = [
        np.mean(h_channel), np.std(h_channel),
        np.mean(s_channel), np.std(s_channel),
        np.mean(v_channel), np.std(v_channel),
    ]
    features.append(np.array(hsv_moments))

    # -------------------------------------------------------------
    # 3. Texture Analysis - Local Binary Patterns (LBP)
    # -------------------------------------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Fine Scale LBP (P=8, R=1)
    lbp_fine = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist_lbp_fine, _ = np.histogram(lbp_fine.ravel(), bins=np.arange(0, 11), range=(0, 10), density=True)
    features.append(hist_lbp_fine)

    # Coarse Scale LBP (P=16, R=2)
    lbp_coarse = local_binary_pattern(gray, P=16, R=2, method='uniform')
    hist_lbp_coarse, _ = np.histogram(lbp_coarse.ravel(), bins=np.arange(0, 19), range=(0, 18), density=True)
    features.append(hist_lbp_coarse)

    # -------------------------------------------------------------
    # 4. Texture Analysis - GLCM Properties
    # -------------------------------------------------------------
    # Quantize grayscale to 16 levels for fast GLCM computation
    gray_quantized = (gray // 16).astype(np.uint8)
    glcm = graycomatrix(gray_quantized, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    
    glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
    glcm_dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    glcm_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    glcm_energy = graycoprops(glcm, 'energy')[0, 0]
    glcm_correlation = graycoprops(glcm, 'correlation')[0, 0]

    glcm_feats = np.array([glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation])
    features.append(glcm_feats)

    # Concatenate all into a single 1D numpy array
    feature_vector = np.concatenate(features).astype(np.float32)
    return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features_batch(file_paths: List[str]) -> np.ndarray:
    """
    Extracts features for a batch of filepaths.
    """
    feature_list = []
    for p in file_paths:
        feat = extract_features_from_image(p)
        feature_list.append(feat)
    return np.array(feature_list)


class ClassicalClassifierPipeline:
    """
    Pipeline encapsulating StandardScaler and Classifiers (SVM / Random Forest).
    """
    def __init__(self, model_type: str = "svm"):
        self.model_type = model_type.lower()
        self.scaler = StandardScaler()
        if self.model_type == "svm":
            self.classifier = SVC(C=10.0, kernel='rbf', probability=True, random_state=42)
        elif self.model_type == "rf":
            self.classifier = RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'svm' or 'rf'.")
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, class_names: Optional[List[str]] = None):
        self.classes_ = class_names if class_names is not None else [str(c) for c in np.unique(y)]
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        report = classification_report(y_test, preds, target_names=self.classes_, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, preds)

        return {
            "model_type": self.model_type,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": cm,
            "classification_report": report
        }

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump({
            "model_type": self.model_type,
            "scaler": self.scaler,
            "classifier": self.classifier,
            "classes_": self.classes_
        }, file_path)

    @classmethod
    def load(cls, file_path: str) -> "ClassicalClassifierPipeline":
        data = joblib.load(file_path)
        inst = cls(model_type=data["model_type"])
        inst.scaler = data["scaler"]
        inst.classifier = data["classifier"]
        inst.classes_ = data["classes_"]
        return inst
