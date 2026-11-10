"""
04_train_resnet50.py
Veritas AI — Phase 4: Reproducible ResNet50 Binary Image Classification Pipeline

Objectives:
- Fine-tune pretrained ResNet50 (torchvision.models.resnet50) for binary veracity classification (REAL vs FAKE).
- Replace the classification head with a regularized binary head.
- Apply standard ImageNet normalization and training data augmentation.
- Train using the EXACT same train/val/test splits as DistilBERT.
- Optimize on train split; select best checkpoint on validation split.
- Save best checkpoint to backend/models/resnet50_baseline.pt.
- Evaluate on untouched test split only after model selection.
- Save structured experiment results under experiments/resnet50_baseline/.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ResNet50Baseline")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "fakeddit"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = ROOT_DIR / "backend" / "models"
EXPERIMENTS_DIR = ROOT_DIR / "experiments" / "resnet50_baseline"

class FakedditImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: Path, transform=None):
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
            # Fallback to zero tensor if image fails to read
            img = torch.zeros((3, 224, 224), dtype=torch.float32)

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
            "id": row.get("id", f"sample_{idx}")
        }

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform

def evaluate_model(model, dataloader, device) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_probs.extend(probs[:, 1])
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)

    # Explicit macro metrics across both classes
    macro_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # Class-specific metrics (Class 0: REAL, Class 1: FAKE)
    real_prec = precision_score(all_labels, all_preds, pos_label=0, average="binary", zero_division=0)
    real_rec = recall_score(all_labels, all_preds, pos_label=0, average="binary", zero_division=0)
    real_f1 = f1_score(all_labels, all_preds, pos_label=0, average="binary", zero_division=0)

    fake_prec = precision_score(all_labels, all_preds, pos_label=1, average="binary", zero_division=0)
    fake_rec = recall_score(all_labels, all_preds, pos_label=1, average="binary", zero_division=0)
    fake_f1 = f1_score(all_labels, all_preds, pos_label=1, average="binary", zero_division=0)

    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        roc_auc = 0.5

    # Specificity = TN / (TN + FP) = Recall of REAL (class 0)
    # FPR = FP / (TN + FP) = 1 - Specificity
    specificity = float(real_rec)
    fpr = float(1.0 - specificity)

    metrics = {
        "loss": round(float(avg_loss), 4),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "real_precision": round(float(real_prec), 4),
        "real_recall": round(float(real_rec), 4),
        "real_f1": round(float(real_f1), 4),
        "fake_precision": round(float(fake_prec), 4),
        "fake_recall": round(float(fake_rec), 4),
        "fake_f1": round(float(fake_f1), 4),
        "f1_score": round(float(macro_f1), 4),  # Standardized to Macro F1
        "precision": round(float(macro_prec), 4),
        "recall": round(float(macro_rec), 4),
        "roc_auc": round(float(roc_auc), 4),
        "specificity": round(float(specificity), 4),
        "false_positive_rate": round(float(fpr), 4)
    }
    return metrics, all_preds, all_labels

def build_resnet50_model(num_classes: int = 2, freeze_early_layers: bool = True):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    if freeze_early_layers:
        # Freeze conv1, bn1, layer1, layer2; fine-tune layer3, layer4, and fc
        for param in list(model.parameters())[:-30]:
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
    return model

def train_resnet50_pipeline(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,
    seed: int = 42
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("Veritas AI — Phase 4: ResNet50 Binary Image Classification Training")
    logger.info(f"Device: {device} | Seed: {seed} | Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
    logger.info("=" * 70)

    # 1. Load splits (exact same split files as text model)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    logger.info(f"Dataset loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Transforms & Datasets
    train_trans, eval_trans = get_transforms()
    train_ds = FakedditImageDataset(train_df, IMAGES_DIR, transform=train_trans)
    val_ds = FakedditImageDataset(val_df, IMAGES_DIR, transform=eval_trans)
    test_ds = FakedditImageDataset(test_df, IMAGES_DIR, transform=eval_trans)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Model & Optimizer
    model = build_resnet50_model(num_classes=2, freeze_early_layers=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 4. Training loop
    best_val_f1 = -1.0
    best_checkpoint_path = MODELS_DIR / "resnet50_baseline.pt"
    training_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                logger.info(f"Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_ds)
        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        elapsed = time.time() - t0

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(float(avg_train_loss), 4),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1_score"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_roc_auc": val_metrics["roc_auc"],
            "epoch_duration_sec": round(elapsed, 1)
        }
        training_history.append(epoch_record)
        logger.info(f"Epoch {epoch} Completed ({elapsed:.1f}s): Train Loss={avg_train_loss:.4f} | Val F1={val_metrics['f1_score']} | Val Acc={val_metrics['accuracy']}")

        # Model selection on validation F1
        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"  -> Saved new best checkpoint (Val F1: {best_val_f1}) to {best_checkpoint_path}")

    # 5. Final Evaluation on UNTOUCHED Test Split
    logger.info("=" * 70)
    logger.info("Loading best selected checkpoint for final evaluation on untouched Test Split...")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    test_metrics, test_preds, test_labels = evaluate_model(model, test_loader, device)

    cm = confusion_matrix(test_labels, test_preds).tolist()
    cls_report = classification_report(test_labels, test_preds, target_names=["REAL (0)", "FAKE (1)"], output_dict=True)

    logger.info("=" * 70)
    logger.info("ResNet50 Final Test Set Results:")
    logger.info(f"  Test Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"  Test F1-Score:  {test_metrics['f1_score'] * 100:.2f}%")
    logger.info(f"  Test Precision: {test_metrics['precision'] * 100:.2f}%")
    logger.info(f"  Test Recall:    {test_metrics['recall'] * 100:.2f}%")
    logger.info(f"  Test ROC-AUC:   {test_metrics['roc_auc'] * 100:.2f}%")
    logger.info(f"  Confusion Matrix: {cm}")
    logger.info("=" * 70)

    # 6. Save Structured Experiment Record
    experiment_record = {
        "model_name": "ResNet50 Baseline (torchvision.models.resnet50)",
        "modality": "image-only",
        "training_environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "random_seed": seed
        },
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "AdamW",
            "weight_decay": 0.01
        },
        "splits": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df)
        },
        "training_history": training_history,
        "best_validation_metrics": [h for h in training_history if h["val_f1"] == best_val_f1][0],
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "checkpoint_path": str(best_checkpoint_path)
    }

    record_path = EXPERIMENTS_DIR / "experiment_record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(experiment_record, f, indent=2)

    logger.info(f"Experiment record saved to: {record_path}")
    return experiment_record

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet50 Training Script")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_resnet50_pipeline(args.epochs, args.batch_size, args.lr, args.seed)
