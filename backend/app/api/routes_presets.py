from fastapi import APIRouter, HTTPException
from typing import List
import json
from backend.app.schemas.response_schemas import DemoPresetResponse
from backend.app.config import settings

router = APIRouter(tags=["Presets"])

@router.get("/presets", response_model=List[DemoPresetResponse])
def get_presets():
    manifest_path = settings.DEMO_DIR / "demo_manifest.json"
    if not manifest_path.exists():
        return []
    
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            presets_data = json.load(f)

        for p in presets_data:
            if not p.get("image_url") and p.get("image_file"):
                p["image_url"] = f"/demo_images/{p['image_file']}"

        return presets_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read demo manifest: {str(e)}")
