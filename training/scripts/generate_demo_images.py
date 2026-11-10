import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def generate_demo_images():
    root_dir = Path(__file__).resolve().parent.parent.parent
    manifest_path = root_dir / "data" / "demo_curated" / "demo_manifest.json"
    images_dir = root_dir / "data" / "demo_curated" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Color palettes for categories
    category_colors = {
        "Science & Astronomy": (15, 23, 42),      # Deep navy
        "History & Exploration": (67, 56, 202),   # Indigo
        "Culture & Events": (180, 83, 9),         # Amber
        "Earth Science & Geology": (185, 28, 28), # Red
        "Politics & Governance": (30, 41, 59),    # Slate
        "Health Misinformation": (159, 18, 57),   # Rose
        "Conspiracy & Pseudohistory": (88, 28, 135), # Purple
        "Out-of-Context Media": (194, 65, 12),    # Orange
        "Pseudoscience Satire": (13, 148, 136),   # Teal
        "Financial Hoax": (51, 65, 85),           # Slate gray
        "Disaster Hoax": (127, 29, 29),           # Crimson
        "Space Exploration": (24, 24, 27),        # Zinc
        "Space & Earth Observation": (30, 58, 138),# Blue
        "Archaeology & History": (120, 53, 15),   # Brown
        "Climate Hyperbole": (22, 101, 52),       # Emerald
        "Technological Conspiracy": (71, 85, 105),# Slate
        "Marine Biology & Conservation": (3, 105, 161), # Sky blue
        "Thermodynamic Hoax": (136, 19, 55),      # Pink/Burgundy
        "Zoology & Extinction": (87, 83, 78)      # Warm stone
    }

    count = 0
    for item in manifest:
        filename = item["image_file"]
        file_path = images_dir / filename
        
        # Determine background color
        cat = item.get("category", "")
        bg_color = category_colors.get(cat, (30, 41, 59))
        
        # Create 224x224 RGB image
        img = Image.new("RGB", (224, 224), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Draw decorative geometric elements
        draw.rectangle([10, 10, 214, 214], outline=(255, 255, 255, 80), width=1)
        draw.rectangle([14, 14, 210, 210], outline=(255, 255, 255, 40), width=1)
        
        # Draw badge for Ground Truth
        gt = item["ground_truth"]
        badge_color = (16, 185, 129) if gt == "REAL" else (244, 63, 94)
        draw.rectangle([20, 20, 90, 42], fill=badge_color)
        draw.text((26, 24), gt, fill=(255, 255, 255))
        
        # Draw category & ID text
        item_id = item["id"]
        draw.text((20, 55), item_id, fill=(203, 213, 225))
        
        # Truncated category label
        draw.text((20, 75), cat[:24], fill=(226, 232, 240))
        
        # Draw central motif (concentric circles/rectangles)
        draw.ellipse([72, 100, 152, 180], outline=(255, 255, 255), width=2)
        draw.text((95, 132), "DEMO", fill=(255, 255, 255))

        # Save with high quality
        img.save(file_path, "JPEG", quality=95)
        count += 1

    print(f"Successfully generated {count} verified demo sample images in: {images_dir}")

if __name__ == "__main__":
    generate_demo_images()
