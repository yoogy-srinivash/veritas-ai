"""
01_download_fakeddit.py
Fakeddit Multimodal Benchmark Acquisition & Validation Pipeline

Objectives:
- Acquire multimodal paired samples from the Fakeddit benchmark (Nakamura et al., 2020).
- Target: 5,000 paired multimodal samples (balanced: 2,500 REAL, 2,500 FAKE), subject to link availability.
- Strict Rule: Missing or corrupt images are replaced ONLY by querying subsequent valid Fakeddit samples.
  Zero synthetic or external images are permitted in data/fakeddit/.
- Image Verification: Verify integrity using PIL.Image.open().verify(). Discard corrupt, HTML, or <64x64 images.
- Documentation: Log actual acquired count, class balance, and link decay statistics.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
import requests
from PIL import Image
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FakedditAcquisition")

def verify_image(file_path: Path, min_dimension: int = 64) -> bool:
    """Validate image file integrity and minimum resolution using Pillow."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Re-open to inspect dimensions (verify() closes file and clears metadata)
        with Image.open(file_path) as img:
            w, h = img.size
            if w < min_dimension or h < min_dimension:
                return False
            # Check for standard RGB / RGBA / L modes
            if img.mode not in ("RGB", "RGBA", "L"):
                return False
        return True
    except Exception:
        return False

def download_image_with_retry(
    url: str,
    dest_path: Path,
    timeout: int = 8,
    max_retries: int = 2
) -> bool:
    """Download single image with custom headers, timeout, and verification."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 VeritasAI/1.0"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "")
                if "image" not in content_type and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    return False
                
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Check file size (> 2KB to filter empty or placeholder icons)
                if dest_path.stat().st_size < 2048:
                    if dest_path.exists():
                        dest_path.unlink()
                    return False
                
                # Validate image integrity
                if verify_image(dest_path):
                    return True
                else:
                    if dest_path.exists():
                        dest_path.unlink()
                    return False
        except Exception:
            pass
        time.sleep(0.5)

    if dest_path.exists():
        try:
            dest_path.unlink()
        except Exception:
            pass
    return False

def acquire_fakeddit_benchmark(
    tsv_path: Optional[str] = None,
    target_per_class: int = 2500,
    output_dir: Optional[str] = None
) -> Tuple[int, int]:
    """
    Acquires up to target_per_class REAL and target_per_class FAKE samples.
    """
    root_dir = Path(__file__).resolve().parent.parent.parent
    if output_dir:
        dest_dir = Path(output_dir)
    else:
        dest_dir = root_dir / "data" / "fakeddit"

    images_dir = dest_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Veritas AI: Fakeddit Multimodal Benchmark Acquisition Pipeline")
    logger.info(f"Target per class: {target_per_class} (Total target: {target_per_class * 2})")
    logger.info(f"Output directory: {dest_dir}")
    logger.info("=" * 60)

    # Check for input TSV or HuggingFace fallback
    df = None
    if tsv_path and Path(tsv_path).exists():
        logger.info(f"Loading local Fakeddit TSV: {tsv_path}")
        df = pd.read_csv(tsv_path, sep="\t" if tsv_path.endswith(".tsv") else ",")
    else:
        # Check if local fakeddit candidates file exists
        local_candidates = dest_dir / "fakeddit_candidates.tsv"
        if local_candidates.exists():
            logger.info(f"Loading local candidates: {local_candidates}")
            df = pd.read_csv(local_candidates, sep="\t")
        else:
            logger.warning("No local Fakeddit candidate TSV found at data/fakeddit/fakeddit_candidates.tsv.")
            logger.info(
                "In Google Colab, candidate URLs are streamed from the official Fakeddit repository "
                "or downloaded via: huggingface-cli / official Google Drive Fakeddit dump."
            )
            return 0, 0

    # Ensure required columns
    required_cols = {"clean_title", "image_url", "2_way_label"}
    if not required_cols.issubset(set(df.columns)):
        # Handle variations in column names (e.g. title vs clean_title)
        col_map = {"title": "clean_title", "label": "2_way_label", "url": "image_url"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Separate into REAL and FAKE pools
    df_real = df[df["2_way_label"] == 0].copy()
    df_fake = df[df["2_way_label"] == 1].copy()

    logger.info(f"Candidate pool: {len(df_real)} REAL candidates, {len(df_fake)} FAKE candidates.")

    verified_samples = []
    stats = {"real_acquired": 0, "fake_acquired": 0, "404_or_corrupt": 0}

    for label_name, label_val, candidate_pool in [("REAL", 0, df_real), ("FAKE", 1, df_fake)]:
        logger.info(f"Processing candidate pool for {label_name} (target: {target_per_class})...")
        count = 0

        for idx, row in candidate_pool.iterrows():
            if count >= target_per_class:
                break

            title = str(row.get("clean_title", "")).strip()
            url = str(row.get("image_url", "")).strip()
            sample_id = str(row.get("id", f"sample_{label_name.lower()}_{count:05d}"))

            if not title or len(title.split()) < 4 or not url.startswith("http"):
                continue

            # Target filename
            ext = ".jpg"
            if ".png" in url.lower():
                ext = ".png"
            elif ".webp" in url.lower():
                ext = ".webp"
            img_filename = f"{sample_id}{ext}"
            img_path = images_dir / img_filename

            # If already downloaded and verified
            if img_path.exists() and verify_image(img_path):
                verified_samples.append({
                    "id": sample_id,
                    "text": title,
                    "image_file": img_filename,
                    "label": label_val,
                    "label_name": label_name,
                    "source_url": url
                })
                count += 1
                stats[f"{label_name.lower()}_acquired"] += 1
                continue

            # Attempt download
            success = download_image_with_retry(url, img_path)
            if success:
                verified_samples.append({
                    "id": sample_id,
                    "text": title,
                    "image_file": img_filename,
                    "label": label_val,
                    "label_name": label_name,
                    "source_url": url
                })
                count += 1
                stats[f"{label_name.lower()}_acquired"] += 1
                if count % 100 == 0:
                    logger.info(f"[{label_name}] Verified {count}/{target_per_class} paired samples...")
            else:
                stats["404_or_corrupt"] += 1

    # Save verified records
    verified_df = pd.DataFrame(verified_samples)
    out_csv = dest_dir / "fakeddit_verified.csv"
    verified_df.to_csv(out_csv, index=False)

    logger.info("=" * 60)
    logger.info("Acquisition Complete!")
    logger.info(f"Verified REAL: {stats['real_acquired']}")
    logger.info(f"Verified FAKE: {stats['fake_acquired']}")
    logger.info(f"Decayed / Corrupt links discarded: {stats['404_or_corrupt']}")
    logger.info(f"Total verified dataset saved to: {out_csv}")
    logger.info("=" * 60)

    return stats["real_acquired"], stats["fake_acquired"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fakeddit Acquisition Script")
    parser.add_argument("--tsv", type=str, default=None, help="Path to raw Fakeddit candidate TSV")
    parser.add_argument("--target", type=int, default=2500, help="Target samples per class")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    acquire_fakeddit_benchmark(args.tsv, args.target, args.out)
