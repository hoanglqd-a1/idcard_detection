import base64
from io import BytesIO
from time import perf_counter

import numpy as np
from PIL import Image, UnidentifiedImageError

from training.detection import CARD_SIZE
from training.utils.processing import load_image

from ..config import Settings
from ..inference.pipeline import Pipeline
from ..schemas.analysis import AnalyzeResponse, Point


class AnalysisError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def decode_image(data: bytes, settings: Settings) -> tuple[np.ndarray, int, int, int]:
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {'JPEG', 'PNG', 'WEBP'}:
                raise AnalysisError(415, 'unsupported_format', 'Use a JPEG, PNG, or WebP image.')
            width, height = source.size
            if width * height > settings.max_image_pixels:
                raise AnalysisError(413, 'image_too_large', 'The decoded image exceeds the pixel limit.')
            if min(width, height) < 2:
                raise AnalysisError(422, 'invalid_image', 'The image must be at least 2 × 2 pixels.')
            if getattr(source, 'n_frames', 1) != 1:
                raise AnalysisError(415, 'animated_image', 'Upload a single still image.')
            source.verify()
        with Image.open(BytesIO(data)) as source:
            # Match legacy input resizing and channel handling exactly.
            working = load_image(BytesIO(data), CARD_SIZE)
            orientation = source.getexif().get(274, 1)
        return working, width, height, orientation
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise AnalysisError(413, 'image_too_large', 'The decoded image exceeds the pixel limit.') from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise AnalysisError(422, 'invalid_image', 'The file is not a valid, complete image.') from exc


def display_point(x: float, y: float, width: int, height: int, orientation: int) -> Point:
    """Map raw raster coordinates to the EXIF-oriented browser preview."""
    x, y = float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))
    transforms = {
        2: (width - 1 - x, y), 3: (width - 1 - x, height - 1 - y),
        4: (x, height - 1 - y), 5: (y, x), 6: (height - 1 - y, x),
        7: (height - 1 - y, width - 1 - x), 8: (y, width - 1 - x),
    }
    x, y = transforms.get(orientation, (x, y))
    return Point(x=x, y=y)


def analyze_image(data: bytes, pipeline: Pipeline, settings: Settings) -> AnalyzeResponse:
    started = perf_counter()
    working, width, height, orientation = decode_image(data, settings)
    result = pipeline.predict(working)
    corners = None
    if result.corners is not None:
        corners = [display_point(x * width / CARD_SIZE[0], y * height / CARD_SIZE[1],
                                 width, height, orientation) for x, y in result.corners]
    extracted = None
    if result.card is not None:
        output = BytesIO()
        Image.fromarray(result.card).save(output, format='PNG')
        extracted = 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii')
    identity = pipeline.templates[result.template_index] if result.template_index is not None else None
    display_width, display_height = (height, width) if orientation in (5, 6, 7, 8) else (width, height)
    return AnalyzeResponse(
        status=result.status, card_detected=result.card_detected,
        template_id=identity.id if identity else None, template_name=identity.name if identity else None,
        detection_confidence=result.detection_confidence, match_score=result.match_score,
        match_threshold=settings.match_threshold, is_supported=result.is_supported,
        corners=corners, image_width=display_width, image_height=display_height,
        extracted_card=extracted, failure_stage=result.failure_stage,
        processing_time_ms=(perf_counter() - started) * 1000,
        inference_time_ms=result.processing_time_ms,
    )
