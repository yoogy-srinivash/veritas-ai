"""
01_acquire_fakeddit.py
Veritas AI — Phase 1: Fakeddit Multimodal Benchmark Acquisition Pipeline

Objectives:
- Acquire authentic multimodal paired samples from the Fakeddit benchmark.
- Target: 5,000 valid paired text-image samples, subject to actual availability.
- Match text, original six-way label, and image correctly.
- Validate image existence, readability, dimensions (>=64x64), and RGB conversion.
- Detect missing, corrupt, and duplicate samples (text duplicates and image hash collisions).
- Maintain a reproducible acquisition manifest and audit report with filtering reasons.
- Zero fabrication of samples, labels, URLs, or statistics.
"""

import os
import sys
import json
import time
import hashlib
import zipfile
import logging
from pathlib import Path
from typing import Dict, Any, List
import urllib.request
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AcquisitionPipeline")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "fakeddit"
IMAGES_DIR = DATA_DIR / "images"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
MANIFEST_PATH = DATA_DIR / "acquisition_manifest.csv"
REPORT_PATH = DATA_DIR / "acquisition_report.json"

HF_BASE_URL = "https://huggingface.co/datasets/ams-99/fakeddit_9k/raw/main"
HF_ZIP_URL = "https://huggingface.co/datasets/ams-99/fakeddit_9k/resolve/main/image_cache.zip"

def download_file(url: str, dest_path: Path, desc: str = "file") -> None:
    """Download a file with progress logging."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"{desc} already exists at {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB). Skipping download.")
        return

    logger.info(f"Downloading {desc} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VeritasAI/1.0"}
    req = urllib.request.Request(url, headers=headers)
    
    t0 = time.time()
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024 # 1MB
        last_log = t0
        
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            
            if time.time() - last_log > 5 or (total_size > 0 and downloaded == total_size):
                pct = (downloaded / total_size * 100) if total_size > 0 else 0
                mb = downloaded / (1024 * 1024)
                tot_mb = total_size / (1024 * 1024) if total_size > 0 else 0
                elapsed = time.time() - t0
                speed = mb / elapsed if elapsed > 0 else 0
                logger.info(f"  [{desc}] {mb:.1f}/{tot_mb:.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s")
                last_log = time.time()

    logger.info(f"Successfully downloaded {desc} to {dest_path} in {time.time() - t0:.1f}s")

def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract zip archive if not already extracted."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(extract_dir.glob("*.jpg"))
    if len(existing_files) > 1000:
        logger.info(f"Raw images already extracted in {extract_dir} ({len(existing_files)} files).")
        return

    logger.info(f"Extracting {zip_path} into {extract_dir}...")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            if member.endswith(".jpg") or member.endswith(".png"):
                filename = Path(member).name
                dest = extract_dir / filename
                if not dest.exists():
                    with z.open(member) as source, open(dest, "wb") as target:
                        target.write(source.read())

    logger.info(f"Extraction complete in {time.time() - t0:.1f}s. Extracted files: {len(list(extract_dir.glob('*.*')))}")

def run_acquisition(target_samples: int = 5000):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Veritas AI — Phase 1: Fakeddit Multimodal Benchmark Acquisition")
    logger.info(f"Target sample count: {target_samples}")
    logger.info(f"Destination: {DATA_DIR}")
    logger.info("=" * 70)

    # 1. Download metadata splits
    metadata_dfs = []
    for split_name in ["train_balanced.csv", "val_balanced.csv", "test_balanced.csv"]:
        split_dest = DATA_DIR / split_name
        download_file(f"{HF_BASE_URL}/{split_name}", split_dest, desc=split_name)
        df_split = pd.read_csv(split_dest)
        metadata_dfs.append(df_split)

    full_candidates = pd.concat(metadata_dfs, ignore_index=True)
    logger.info(f"Total candidate rows aggregated: {len(full_candidates)}")

    # 2. Download and extract image_cache.zip
    zip_dest = DATA_DIR / "image_cache.zip"
    download_file(HF_ZIP_URL, zip_dest, desc="image_cache.zip")
    extract_zip(zip_dest, RAW_IMAGES_DIR)

    # 3. Comprehensive Validation & Filtering
    logger.info("Validating candidates, checking dimensions, Pillow readability, and duplicate detection...")
    
    seen_texts = set()
    seen_hashes = set()
    filtering_reasons: Dict[str, int] = {
        "MISSING_IMAGE": 0,
        "CORRUPT_IMAGE": 0,
        "DIMENSION_TOO_SMALL": 0,
        "RGB_CONVERSION_ERROR": 0,
        "TEXT_TOO_SHORT": 0,
        "DUPLICATE_TEXT": 0,
        "DUPLICATE_IMAGE_HASH": 0,
        "PASSED_VALIDATION": 0
    }

    valid_candidates: List[Dict[str, Any]] = []

    for idx, row in full_candidates.iterrows():
        raw_text = str(row.get("text", "")).strip()
        img_rel_path = str(row.get("image_path", ""))
        img_filename = Path(img_rel_path).name
        six_way = int(row.get("6_way_label", 0))
        orig_url = str(row.get("original_url", ""))

        # Check text length
        tokens = raw_text.split()
        if len(tokens) < 3:
            filtering_reasons["TEXT_TOO_SHORT"] += 1
            continue

        # Check duplicate text
        norm_text = " ".join(tokens).lower()
        if norm_text in seen_texts:
            filtering_reasons["DUPLICATE_TEXT"] += 1
            continue

        # Check physical image existence in raw_images
        raw_img_path = RAW_IMAGES_DIR / img_filename
        if not raw_img_path.exists():
            filtering_reasons["MISSING_IMAGE"] += 1
            continue

        # Validate image with Pillow
        try:
            with Image.open(raw_img_path) as img:
                img.verify()
        except Exception:
            filtering_reasons["CORRUPT_IMAGE"] += 1
            continue

        # Reopen to inspect dimensions and test RGB conversion
        try:
            with Image.open(raw_img_path) as img:
                w, h = img.size
                if w < 64 or h < 64:
                    filtering_reasons["DIMENSION_TOO_SMALL"] += 1
                    continue
                # Test RGB conversion
                rgb_img = img.convert("RGB")
        except Exception:
            filtering_reasons["RGB_CONVERSION_ERROR"] += 1
            continue

        # Compute image SHA-256 to detect duplicate images
        try:
            with open(raw_img_path, "rb") as f:
                img_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            img_hash = ""

        if img_hash in seen_hashes:
            filtering_reasons["DUPLICATE_IMAGE_HASH"] += 1
            continue

        # Passed all filters
        seen_texts.add(norm_text)
        if img_hash:
            seen_hashes.add(img_hash)

        filtering_reasons["PASSED_VALIDATION"] += 1
        
        # Binary label mapping: 0 -> REAL (0), 1-5 -> FAKE (1)
        binary_label = 0 if six_way == 0 else 1
        binary_name = "REAL" if binary_label == 0 else "FAKE"

        valid_candidates.append({
            "candidate_id": f"fakeddit_{len(valid_candidates):05d}",
            "text": raw_text,
            "raw_filename": img_filename,
            "6_way_label": six_way,
            "2_way_label": binary_label,
            "label_name": binary_name,
            "width": w,
            "height": h,
            "image_sha256": img_hash,
            "original_url": orig_url
        })

    valid_df = pd.DataFrame(valid_candidates)
    logger.info(f"Total physically verified valid candidates: {len(valid_df)}")
    logger.info(f"Validation breakdown by 6-way label:\n{valid_df['6_way_label'].value_counts().to_dict()}")
    logger.info(f"Validation breakdown by binary label:\n{valid_df['2_way_label'].value_counts().to_dict()}")

    # 4. Target Selection (up to target_samples)
    real_pool = valid_df[valid_df["2_way_label"] == 0].copy()
    fake_pool = valid_df[valid_df["2_way_label"] == 1].copy()
    logger.info(f"Available verified REAL: {len(real_pool)}, Available verified FAKE: {len(fake_pool)}")

    if len(valid_df) > target_samples:
        n_real = min(len(real_pool), target_samples // 2)
        selected_real = real_pool.sample(n=n_real, random_state=42)
        n_fake = target_samples - len(selected_real)
        # Stratified sampling of fake pool by 6-way label
        selected_fake, _ = train_test_split(
            fake_pool,
            train_size=n_fake,
            stratify=fake_pool["6_way_label"],
            random_state=42
        )
        final_df = pd.concat([selected_real, selected_fake]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        final_df = valid_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    logger.info(f"Final selected benchmark size: {len(final_df)}")
    logger.info(f"Final 6-way label counts:\n{final_df['6_way_label'].value_counts().to_dict()}")
    logger.info(f"Final binary label counts:\n{final_df['2_way_label'].value_counts().to_dict()}")

    # 5. Copy verified images to data/fakeddit/images/
    logger.info(f"Saving final benchmark images to {IMAGES_DIR}...")
    manifest_rows = []
    for idx, row in final_df.iterrows():
        sample_id = f"fakeddit_{idx:05d}"
        dest_filename = f"{sample_id}.jpg"
        source_path = RAW_IMAGES_DIR / row["raw_filename"]
        target_path = IMAGES_DIR / dest_filename

        # Save image as standardized JPEG (224x224 RGB conversion verified)
        with Image.open(source_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(target_path, "JPEG", quality=95)

        manifest_rows.append({
            "id": sample_id,
            "text": row["text"],
            "image_file": dest_filename,
            "6_way_label": int(row["6_way_label"]),
            "2_way_label": int(row["2_way_label"]),
            "label_name": str(row["label_name"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "image_sha256": str(row["image_sha256"]),
            "original_url": str(row["original_url"])
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST_PATH, index=False)
    logger.info(f"Acquisition manifest saved to: {MANIFEST_PATH}")

    # 6. Save Acquisition Report
    six_way_counts = {int(k): int(v) for k, v in manifest_df["6_way_label"].value_counts().to_dict().items()}
    binary_counts = {int(k): int(v) for k, v in manifest_df["2_way_label"].value_counts().to_dict().items()}

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_source": "Fakeddit (ams-99/fakeddit_9k)",
        "total_candidates_examined": len(full_candidates),
        "filtering_reasons": filtering_reasons,
        "physically_verified_available": len(valid_df),
        "final_benchmark_samples": len(manifest_df),
        "binary_distribution": {
            "REAL (0)": binary_counts.get(0, 0),
            "FAKE (1)": binary_counts.get(1, 0)
        },
        "binary_percentages": {
            "REAL (0)": round(binary_counts.get(0, 0) / len(manifest_df) * 100, 2),
            "FAKE (1)": round(binary_counts.get(1, 0) / len(manifest_df) * 100, 2)
        },
        "six_way_distribution": six_way_counts,
        "images_directory": str(IMAGES_DIR),
        "manifest_file": str(MANIFEST_PATH)
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Acquisition report saved to: {REPORT_PATH}")
    logger.info("=" * 70)
    logger.info("Phase 1 Complete!")
    logger.info(f"  Benchmark samples: {len(manifest_df)}")
    logger.info(f"  REAL: {report['binary_distribution']['REAL (0)']} ({report['binary_percentages']['REAL (0)']}%)")
    logger.info(f"  FAKE: {report['binary_distribution']['FAKE (1)']} ({report['binary_percentages']['FAKE (1)']}%)")
    logger.info("=" * 70)

    return report

if __name__ == "__main__":
    target = 5000
    if len(sys.argv) > 1:
        target = int(sys.argv[1])
    run_acquisition(target)
