from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True)

    PROJECT_NAME: str = "Veritas AI"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"
    
    # Path configuration
    APP_DIR: Path = Path(__file__).resolve().parent
    BACKEND_DIR: Path = APP_DIR.parent
    ROOT_DIR: Path = BACKEND_DIR.parent
    
    DATA_DIR: Path = ROOT_DIR / "data"
    DEMO_DIR: Path = DATA_DIR / "demo_curated"
    FAKEDDIT_DIR: Path = DATA_DIR / "fakeddit"
    MODELS_DIR: Path = BACKEND_DIR / "models"
    
    # SQLite Database
    SQLITE_DB_PATH: Path = BACKEND_DIR / "veritas.db"
    DATABASE_URL: str = f"sqlite:///{SQLITE_DB_PATH.as_posix()}"
    
    # Default Inference & Calibration Settings (updated dynamically from calibration_params.json)
    DEFAULT_TEMPERATURE: float = 1.0
    DEFAULT_REJECTION_THRESHOLD: float = 0.70
    DEVICE: str = "cpu"
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

settings = Settings()
