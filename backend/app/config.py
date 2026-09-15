from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='IDCARD_', env_file='.env', extra='ignore')

    detector_path: Path = ROOT / 'training/model/yolov8s-detect.pt'
    face_model_path: Path = ROOT / 'training/model/yolov8n-face.pt'
    template_dir: Path = ROOT / 'training/template_samples'
    device: str = 'cpu'
    match_threshold: float = Field(default=0.8, ge=-1, le=1)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=20_000_000, gt=0)
