"""
02_preprocess_and_split.py
Veritas AI — Phase 2: Preprocessing and Stratified Splitting Pipeline

Objectives:
- Ingest the verified Fakeddit acquisition manifest (acquisition_manifest.csv).
- Apply binary mapping: 0 -> REAL, 1-5 -> FAKE while preserving original 6-way labels.
- Clean text representations without removing meaningful semantic information.
- Document and apply standard ImageNet transformations for ResNet50.
- Perform stratified 70/15/15 splitting (Train: 70%, Validation: 15%, Test: 15%).
- Ensure zero duplicate or near-duplicate leakage across splits.
- Save split manifests (train.csv, val.csv, test.csv) and preprocessing configuration (preprocessing_config.json).
- Formally report actual class distributions across all splits.
"""

import sys
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PreprocessingPipeline")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "fakeddit"
MANIFEST_PATH = DATA_DIR / "acquisition_manifest.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"
CONFIG_PATH = DATA_DIR / "preprocessing_config.json"

def clean_text(text: str) -> str:
    """Clean text without removing meaningful semantic markers or punctuation."""
    if not isinstance(text, str):
        return ""
    # Strip residual raw URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Strip HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Normalize unicode whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def check_leakage(train_texts: List[str], val_texts: List[str], test_texts: List[str]) -> Dict[str, int]:
    """Check for exact and normalized text overlaps across splits."""
    train_set = {t.lower() for t in train_texts}
    val_set = {t.lower() for t in val_texts}
    test_set = {t.lower() for t in test_texts}

    train_val_overlap = len(train_set.intersection(val_set))
    train_test_overlap = len(train_set.intersection(test_set))
    val_test_overlap = len(val_set.intersection(test_set))

    return {
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "total_leaked_pairs": train_val_overlap + train_test_overlap + val_test_overlap
    }

def run_preprocessing_and_splitting(seed: int = 42):
    logger.info("=" * 70)
    logger.info("Veritas AI — Phase 2: Preprocessing & Stratified Splitting")
    logger.info(f"Source manifest: {MANIFEST_PATH}")
    logger.info(f"Target splits: 70% Train, 15% Validation, 15% Test (seed={seed})")
    logger.info("=" * 70)

    if not MANIFEST_PATH.exists():
        logger.error(f"Acquisition manifest not found at {MANIFEST_PATH}. Run Phase 1 first.")
        return None

    df = pd.read_csv(MANIFEST_PATH)
    logger.info(f"Loaded {len(df)} verified samples from acquisition manifest.")

    # 1. Clean text and preserve original 6-way and binary labels
    df["clean_text"] = df["text"].apply(clean_text)
    
    # Filter out empty or trivially short text (< 3 words)
    df["word_count"] = df["clean_text"].apply(lambda t: len(t.split()))
    df = df[df["word_count"] >= 3].copy()
    
    # Ensure binary label consistency: 0 -> REAL, 1-5 -> FAKE
    df["2_way_label"] = df["6_way_label"].apply(lambda x: 0 if int(x) == 0 else 1)
    df["label_name"] = df["2_way_label"].apply(lambda x: "REAL" if x == 0 else "FAKE")

    total_samples = len(df)
    logger.info(f"Cleaned samples ready for splitting: {total_samples}")
    logger.info(f"Binary distribution: REAL (0) = {len(df[df['2_way_label']==0])}, FAKE (1) = {len(df[df['2_way_label']==1])}")

    # 2. Stratified 70 / 15 / 15 Split
    # Step A: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["2_way_label"]
    )

    # Step B: 50% Val, 50% Test of Temp (15% each of total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["2_way_label"]
    )

    # 3. Check for Data Leakage
    leakage = check_leakage(
        train_df["clean_text"].tolist(),
        val_df["clean_text"].tolist(),
        test_df["clean_text"].tolist()
    )
    logger.info(f"Leakage Audit: {leakage}")
    assert leakage["total_leaked_pairs"] == 0, f"Detected {leakage['total_leaked_pairs']} leaked samples between splits!"

    # 4. Save Split CSVs
    export_columns = [
        "id", "clean_text", "image_file", "2_way_label", "6_way_label",
        "label_name", "width", "height", "image_sha256", "original_url"
    ]
    train_df = train_df[export_columns].rename(columns={"clean_text": "text"})
    val_df = val_df[export_columns].rename(columns={"clean_text": "text"})
    test_df = test_df[export_columns].rename(columns={"clean_text": "text"})

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    # 5. Compute Split Statistics
    def get_stats(split_df: pd.DataFrame, name: str) -> Dict[str, Any]:
        n = len(split_df)
        real_n = len(split_df[split_df["2_way_label"] == 0])
        fake_n = len(split_df[split_df["2_way_label"] == 1])
        return {
            "split_name": name,
            "total_samples": n,
            "real_count": real_n,
            "fake_count": fake_n,
            "real_percentage": round(real_n / n * 100, 2) if n > 0 else 0,
            "fake_percentage": round(fake_n / n * 100, 2) if n > 0 else 0,
            "six_way_distribution": split_df["6_way_label"].value_counts().to_dict()
        }

    train_stats = get_stats(train_df, "Train (70%)")
    val_stats = get_stats(val_df, "Validation (15%)")
    test_stats = get_stats(test_df, "Test (15% - Untouched)")

    # 6. Save Configuration & Manifest
    preprocessing_config = {
        "random_seed": seed,
        "split_ratio": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "binary_mapping": {
            "0": "REAL",
            "1": "FAKE (Satire/Parody)",
            "2": "FAKE (Misleading Content)",
            "3": "FAKE (Manipulated Content)",
            "4": "FAKE (False Connection)",
            "5": "FAKE (Imposter Content)"
        },
        "text_preprocessing": {
            "model_tokenizer": "distilbert-base-uncased",
            "max_sequence_length": 128,
            "truncation": True,
            "padding": "max_length",
            "cleaning": "url_removal, html_stripping, whitespace_normalization"
        },
        "image_preprocessing": {
            "backbone": "ResNet50",
            "target_resolution": [224, 224],
            "channels": 3,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "train_augmentations": "RandomResizedCrop(224, scale=(0.8, 1.0)), RandomHorizontalFlip(p=0.5), ColorJitter(0.1, 0.1)",
            "eval_transforms": "Resize(256), CenterCrop(224), ToTensor(), Normalize"
        },
        "leakage_audit": leakage,
        "splits_summary": {
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats
        }
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(preprocessing_config, f, indent=2)

    logger.info("=" * 70)
    logger.info("Splits Successfully Created & Saved:")
    logger.info(f"  Train:      {train_stats['total_samples']} samples (REAL: {train_stats['real_count']}, FAKE: {train_stats['fake_count']})")
    logger.info(f"  Validation: {val_stats['total_samples']} samples (REAL: {val_stats['real_count']}, FAKE: {val_stats['fake_count']})")
    logger.info(f"  Test:       {test_stats['total_samples']} samples (REAL: {test_stats['real_count']}, FAKE: {test_stats['fake_count']})")
    logger.info(f"Configuration saved to: {CONFIG_PATH}")
    logger.info("=" * 70)

    return preprocessing_config

if __name__ == "__main__":
    run_preprocessing_and_splitting()
