"""
Veritas AI — Independent Evaluation Audit & Metric Recalculation Script
Task: Day 2.1 Evaluation Audit and Metric Correction

Independently loads checkpoints, runs inference on the untouched test split (and validation split),
calculates all required metrics with explicit class ordering using scikit-learn,
and diagnoses class imbalance / threshold dynamics.
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "fakeddit"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = BASE_DIR / "backend" / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

class FakedditTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class FakedditImageDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image_file"]
        img_path = self.images_dir / img_name
        label = int(row["2_way_label"])

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
        except Exception:
            img = torch.zeros((3, 224, 224), dtype=torch.float32)

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long)
        }

def compute_all_metrics(y_true, y_pred, y_prob):
    """
    Computes all standard evaluation metrics with explicit class ordering.
    Class 0: REAL (Negative)
    Class 1: FAKE (Positive)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = float(accuracy_score(y_true, y_pred))

    # Macro averages
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Class 0: REAL
    real_prec = float(precision_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0))
    real_rec = float(recall_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0))
    real_f1 = float(f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0))

    # Class 1: FAKE
    fake_prec = float(precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))
    fake_rec = float(recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))
    fake_f1 = float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))

    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    # Confusion matrix with explicit labels [0, 1]
    # cm[0,0] = TN (REAL predicted as REAL)
    # cm[0,1] = FP (REAL predicted as FAKE)
    # cm[1,0] = FN (FAKE predicted as REAL)
    # cm[1,1] = TP (FAKE predicted as FAKE)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Specificity and FPR (considering FAKE as positive class 1)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # Equal to real_rec
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0           # 1 - specificity

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "real_precision": round(real_prec, 4),
        "real_recall": round(real_rec, 4),
        "real_f1": round(real_f1, 4),
        "fake_precision": round(fake_prec, 4),
        "fake_recall": round(fake_rec, 4),
        "fake_f1": round(fake_f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": ["REAL (0)", "FAKE (1)"],
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "specificity": round(specificity, 4),
        "false_positive_rate": round(fpr, 4)
    }

def analyze_probability_distribution(probs, labels, name="Model"):
    real_probs = probs[labels == 0]
    fake_probs = probs[labels == 1]
    print(f"\n--- {name} Probability Distribution Analysis (P(FAKE)) ---")
    print(f"Overall Mean P(FAKE): {np.mean(probs):.4f} (std: {np.std(probs):.4f})")
    print(f"REAL samples (N={len(real_probs)}): Mean P(FAKE) = {np.mean(real_probs):.4f}, Median = {np.median(real_probs):.4f}, Min = {np.min(real_probs):.4f}, Max = {np.max(real_probs):.4f}")
    print(f"FAKE samples (N={len(fake_probs)}): Mean P(FAKE) = {np.mean(fake_probs):.4f}, Median = {np.median(fake_probs):.4f}, Min = {np.min(fake_probs):.4f}, Max = {np.max(fake_probs):.4f}")
    
    # Threshold sweep on validation set
    thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    print("\nDecision Threshold Sweep (on Validation set only):")
    print("Thresh | Acc    | Macro-F1 | REAL-F1 | FAKE-F1 | REAL-Rec | FAKE-Rec | Pred-FAKE%")
    print("-" * 75)
    best_th = 0.50
    best_macro = 0.0
    for th in thresholds:
        preds = (probs >= th).astype(int)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        real_f1 = f1_score(labels, preds, pos_label=0, average="binary", zero_division=0)
        fake_f1 = f1_score(labels, preds, pos_label=1, average="binary", zero_division=0)
        real_rec = recall_score(labels, preds, pos_label=0, average="binary", zero_division=0)
        fake_rec = recall_score(labels, preds, pos_label=1, average="binary", zero_division=0)
        pct_fake = np.mean(preds) * 100
        if macro_f1 > best_macro:
            best_macro = macro_f1
            best_th = th
        print(f" {th:.2f}  | {acc:.4f} |  {macro_f1:.4f}  | {real_f1:.4f}  | {fake_f1:.4f}  |  {real_rec:.4f}  |  {fake_rec:.4f}  |  {pct_fake:5.1f}%")
    print(f"Best Sampled Threshold for Macro F1 on Validation (among evaluated grid): {best_th:.2f} (Macro F1 = {best_macro:.4f})")
    return best_th

def main():
    print("=" * 80)
    print("Veritas AI — Day 2.1 Independent Metric Recalculation & Audit")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load splits
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"\nDataset Splits:")
    print(f"  Train: {len(train_df)} (REAL: {sum(train_df['2_way_label']==0)}, FAKE: {sum(train_df['2_way_label']==1)})")
    print(f"  Val:   {len(val_df)} (REAL: {sum(val_df['2_way_label']==0)}, FAKE: {sum(val_df['2_way_label']==1)})")
    print(f"  Test:  {len(test_df)} (REAL: {sum(test_df['2_way_label']==0)}, FAKE: {sum(test_df['2_way_label']==1)})")

    # 1. DistilBERT
    print("\n" + "=" * 50)
    print("1. Auditing DistilBERT Text Baseline")
    print("=" * 50)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODELS_DIR / "distilbert_tokenizer")
    text_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    text_model.load_state_dict(torch.load(MODELS_DIR / "distilbert_baseline.pt", map_location=device))
    text_model.to(device)
    text_model.eval()

    def get_text_predictions(df):
        ds = FakedditTextDataset(df["text"].tolist(), df["2_way_label"].tolist(), tokenizer)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"]
                outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.extend(probs[:, 1])
                all_preds.extend(np.argmax(probs, axis=-1))
                all_labels.extend(labels.numpy())
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    print("Running DistilBERT on untouched TEST split...")
    t_test_labels, t_test_preds, t_test_probs = get_text_predictions(test_df)
    distilbert_test_audit = compute_all_metrics(t_test_labels, t_test_preds, t_test_probs)

    print("Running DistilBERT on VALIDATION split...")
    t_val_labels, t_val_preds, t_val_probs = get_text_predictions(val_df)
    distilbert_val_audit = compute_all_metrics(t_val_labels, t_val_preds, t_val_probs)

    # 2. ResNet50
    print("\n" + "=" * 50)
    print("2. Auditing ResNet50 Vision Baseline")
    print("=" * 50)
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    vision_model = models.resnet50()
    num_features = vision_model.fc.in_features
    vision_model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )
    vision_model.load_state_dict(torch.load(MODELS_DIR / "resnet50_baseline.pt", map_location=device))
    vision_model.to(device)
    vision_model.eval()

    def get_image_predictions(df):
        ds = FakedditImageDataset(df, images_dir=IMAGES_DIR, transform=eval_transform)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                labels = batch["label"]
                outputs = vision_model(images)
                probs = torch.softmax(outputs, dim=-1).cpu().numpy()
                all_probs.extend(probs[:, 1])
                all_preds.extend(np.argmax(probs, axis=-1))
                all_labels.extend(labels.numpy())
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    print("Running ResNet50 on untouched TEST split...")
    i_test_labels, i_test_preds, i_test_probs = get_image_predictions(test_df)
    resnet_test_audit = compute_all_metrics(i_test_labels, i_test_preds, i_test_probs)

    print("Running ResNet50 on VALIDATION split...")
    i_val_labels, i_val_preds, i_val_probs = get_image_predictions(val_df)
    resnet_val_audit = compute_all_metrics(i_val_labels, i_val_preds, i_val_probs)

    # Output detailed report
    audit_report = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "positive_class_definition": "Class 1: FAKE (Class 0: REAL is treated as negative/normal)",
        "distilbert": {
            "test_metrics": distilbert_test_audit,
            "validation_metrics": distilbert_val_audit
        },
        "resnet50": {
            "test_metrics": resnet_test_audit,
            "validation_metrics": resnet_val_audit
        }
    }

    audit_out_path = EXPERIMENTS_DIR / "evaluation_audit_day2_1.json"
    with open(audit_out_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"\nAudit saved to {audit_out_path}")

    # Analyze probability distributions & threshold sweeps on validation data
    distilbert_best_th = analyze_probability_distribution(t_val_probs, t_val_labels, "DistilBERT (Validation)")
    resnet_best_th = analyze_probability_distribution(i_val_probs, i_val_labels, "ResNet50 (Validation)")

    print("\n" + "=" * 80)
    print("Summary of Corrected Test Metrics (Untouched Test Set, N=750):")
    print("=" * 80)
    print(f"{'Metric':<25} | {'DistilBERT (Text)':<20} | {'ResNet50 (Image)':<20}")
    print("-" * 70)
    for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
              "real_precision", "real_recall", "real_f1",
              "fake_precision", "fake_recall", "fake_f1",
              "roc_auc", "specificity", "false_positive_rate"]:
        print(f"{k:<25} | {distilbert_test_audit[k]:<20} | {resnet_test_audit[k]:<20}")

    print("\nConfusion Matrices [ [TN (REAL), FP (REAL->FAKE)], [FN (FAKE->REAL), TP (FAKE)] ]:")
    print(f"DistilBERT Test CM: {distilbert_test_audit['confusion_matrix']['matrix']}")
    print(f"ResNet50 Test CM:   {resnet_test_audit['confusion_matrix']['matrix']}")

if __name__ == "__main__":
    main()
