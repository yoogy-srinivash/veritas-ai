"""
02_preprocess_dataset.py
Veritas AI — Stratified Preprocessing & Data Splitting Pipeline

Objectives:
- Ingest verified multimodal Fakeddit pairs from fakeddit_verified.csv.
- Re-validate physical image existence and Pillow readability.
- Clean and normalize text representations.
- Perform stratified splitting:
  - Train: 70%
  - Validation: 15% (reserved strictly for temperature scaling & rejection threshold tuning)
  - Test: 15% (preserved untouched until final evaluation)
- Fixed seed (seed=42) for scientific reproducibility.
"""

import sys
import re
import logging
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DataPreprocessor")

def clean_text(text: str) -> str:
    """Normalize whitespace and strip unprintable characters."""
    if not isinstance(text, str):
        return ""
    # Strip raw URL strings
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove HTML artifact tags
    text = re.sub(r"<.*?>", "", text)
    # Normalize unicode whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_and_split(
    verified_csv_path: Path,
    output_dir: Path,
    seed: int = 42
):
    logger.info("=" * 60)
    logger.info("Veritas AI: Stratified Dataset Splitter & Integrity Check")
    logger.info(f"Verified input CSV: {verified_csv_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Random seed: {seed}")
    logger.info("=" * 60)

    if not verified_csv_path.exists():
        logger.error(f"Input file not found: {verified_csv_path}")
        return

    df = pd.read_csv(verified_csv_path)
    logger.info(f"Loaded {len(df)} records from verified CSV.")

    images_dir = output_dir / "images"
    valid_rows = []

    # Final physical verification
    for idx, row in df.iterrows():
        img_name = str(row["image_file"])
        img_path = images_dir / img_name

        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            continue

        cleaned = clean_text(str(row["text"]))
        if len(cleaned.split()) < 3:
            continue

        valid_rows.append({
            "id": row["id"],
            "text": cleaned,
            "image_file": img_name,
            "label": int(row["label"]),
            "label_name": "REAL" if int(row["label"]) == 0 else "FAKE"
        })

    clean_df = pd.DataFrame(valid_rows)
    total_valid = len(clean_df)
    logger.info(f"Physically validated and cleaned pairs: {total_valid}")

    if total_valid < 10:
        logger.error("Insufficient samples to perform 70/15/15 stratified split.")
        return

    # Check class balance
    real_count = len(clean_df[clean_df["label"] == 0])
    fake_count = len(clean_df[clean_df["label"] == 1])
    logger.info(f"Class distribution: REAL = {real_count} ({real_count/total_valid*100:.1f}%), FAKE = {fake_count} ({fake_count/total_valid*100:.1f}%)")

    # Stratified Split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        clean_df,
        test_size=0.30,
        random_state=seed,
        stratify=clean_df["label"]
    )

    # Split Temp into 50% Val (15% total), 50% Test (15% total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"]
    )

    # Save splits
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("=" * 60)
    logger.info("Dataset Splits Successfully Generated:")
    logger.info(f"  Train Split (70%): {len(train_df)} samples -> {train_path}")
    logger.info(f"  Validation Split (15%): {len(val_df)} samples -> {val_path}")
    logger.info(f"  Test Split (15% - UNTOUCHED): {len(test_df)} samples -> {test_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    data_fakeddit = root / "data" / "fakeddit"
    csv_file = data_fakeddit / "fakeddit_verified.csv"
    preprocess_and_split(csv_file, data_fakeddit)
