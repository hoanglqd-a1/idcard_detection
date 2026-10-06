from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from training.detection import DetectionStatus


class Point(BaseModel):
    x: float
    y: float


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    status: DetectionStatus
    card_detected: bool
    template_id: str | None = None
    template_name: str | None = None
    detection_confidence: float | None = Field(default=None, ge=0, le=1)
    match_score: float | None = Field(default=None, ge=-1, le=1)
    match_threshold: float
    is_supported: bool | None = None
    corners: list[Point] | None = Field(
        default=None,
        description='Refined boundary, or detected OBB if extraction fails, in EXIF-oriented preview pixels',
    )
    image_width: int
    image_height: int
    extracted_card: str | None = Field(default=None, description='PNG data URL, unmasked RGB crop')
    failure_stage: Literal['crop', 'refinement'] | None = None
    processing_time_ms: float = Field(ge=0, description='Decode through encoding, including inference wait; excludes upload transfer')
    inference_time_ms: float = Field(ge=0, description='Existing pipeline wall time, excluding lock wait')


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: Literal['ready', 'not_ready']
    models_loaded: bool
    template_count: int
