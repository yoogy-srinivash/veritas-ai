from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from backend.app.config import settings
from backend.app.core.database import init_db
from backend.app.api import (
    routes_health,
    routes_presets,
    routes_metrics,
    routes_explanations,
    routes_history,
    routes_verify,
)

# Ensure database tables exist immediately
init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Multimodal NLP Content Verification & Uncertainty Quantification System",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for demo images
demo_images_dir = settings.DEMO_DIR / "images"
if demo_images_dir.exists():
    app.mount("/demo_images", StaticFiles(directory=str(demo_images_dir)), name="demo_images")

# Register API Routers
app.include_router(routes_health.router, prefix=settings.API_V1_STR)
app.include_router(routes_presets.router, prefix=settings.API_V1_STR)
app.include_router(routes_metrics.router, prefix=settings.API_V1_STR)
app.include_router(routes_explanations.router, prefix=settings.API_V1_STR)
app.include_router(routes_history.router, prefix=settings.API_V1_STR)
app.include_router(routes_verify.router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_url": f"{settings.API_V1_STR}/health"
    }
