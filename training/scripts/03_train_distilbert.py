"""
03_train_distilbert.py
Veritas AI — Phase 3: Reproducible DistilBERT Binary Classification Pipeline

Objectives:
- Fine-tune pretrained DistilBERT (distilbert-base-uncased) for binary veracity classification (REAL vs FAKE).
- Document and enforce sequence length (max_length = 128 tokens covers >99% of Reddit titles).
- Optimize on train split; select best checkpoint on validation split.
- Save best checkpoint to backend/models/distilbert_baseline.pt and save tokenizer.
- Record epoch-level loss, validation accuracy, F1, precision, recall, and ROC-AUC.
- Evaluate final model on the untouched test split only after model selection.
- Save structured experiment results under experiments/distilbert_baseline/.
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
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
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
logger = logging.getLogger("DistilBertBaseline")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "fakeddit"
MODELS_DIR = ROOT_DIR / "backend" / "models"
EXPERIMENTS_DIR = ROOT_DIR / "experiments" / "distilbert_baseline"

class FakedditTextDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 128):
        self.texts = [str(t) for t in texts]
        self.labels = [int(l) for l in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
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

def evaluate_model(model, dataloader, device) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * len(labels)

            logits = outputs.logits
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
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        roc_auc = 0.5

    metrics = {
        "loss": round(float(avg_loss), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
    }
    return metrics, all_preds, all_labels

def train_distilbert_pipeline(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    seed: int = 42
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("Veritas AI — Phase 3: DistilBERT Binary Classification Training")
    logger.info(f"Device: {device} | Seed: {seed} | Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
    logger.info(f"Sequence length choice: {max_length} tokens (captures >99% of Reddit titles without padding waste)")
    logger.info("=" * 70)

    # 1. Load splits
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    logger.info(f"Dataset loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Tokenizer & Datasets
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_ds = FakedditTextDataset(train_df["text"].tolist(), train_df["2_way_label"].tolist(), tokenizer, max_length)
    val_ds = FakedditTextDataset(val_df["text"].tolist(), val_df["2_way_label"].tolist(), tokenizer, max_length)
    test_ds = FakedditTextDataset(test_df["text"].tolist(), test_df["2_way_label"].tolist(), tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Model initialization
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    # 4. Training loop
    best_val_f1 = -1.0
    best_checkpoint_path = MODELS_DIR / "distilbert_baseline.pt"
    training_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(labels)
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                logger.info(f"Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | Current Batch Loss: {loss.item():.4f}")

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
            # Save tokenizer alongside model
            tokenizer.save_pretrained(MODELS_DIR / "distilbert_tokenizer")
            logger.info(f"  -> Saved new best checkpoint (Val F1: {best_val_f1}) to {best_checkpoint_path}")

    # 5. Final Evaluation on UNTOUCHED Test Split
    logger.info("=" * 70)
    logger.info("Loading best selected checkpoint for final evaluation on untouched Test Split...")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    test_metrics, test_preds, test_labels = evaluate_model(model, test_loader, device)

    cm = confusion_matrix(test_labels, test_preds).tolist()
    cls_report = classification_report(test_labels, test_preds, target_names=["REAL (0)", "FAKE (1)"], output_dict=True)

    logger.info("=" * 70)
    logger.info("DistilBERT Final Test Set Results:")
    logger.info(f"  Test Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"  Test F1-Score:  {test_metrics['f1_score'] * 100:.2f}%")
    logger.info(f"  Test Precision: {test_metrics['precision'] * 100:.2f}%")
    logger.info(f"  Test Recall:    {test_metrics['recall'] * 100:.2f}%")
    logger.info(f"  Test ROC-AUC:   {test_metrics['roc_auc'] * 100:.2f}%")
    logger.info(f"  Confusion Matrix: {cm}")
    logger.info("=" * 70)

    # 6. Save Structured Experiment Record
    experiment_record = {
        "model_name": "DistilBERT Baseline (distilbert-base-uncased)",
        "modality": "text-only",
        "training_environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "random_seed": seed
        },
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_sequence_length": max_length,
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
    parser = argparse.ArgumentParser(description="DistilBERT Training Script")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_distilbert_pipeline(args.epochs, args.batch_size, args.lr, args.max_length, args.seed)
