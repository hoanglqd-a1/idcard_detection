"""Serving adapter for the shared original-image inference pipeline."""

import os
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import numpy as np

from training.detection import IMAGE_EXTENSIONS, DetectionResult, analyze_card, load_template

from ..config import Settings


@dataclass(frozen=True)
class TemplateIdentity:
    id: str
    name: str


class Pipeline(Protocol):
    templates: list[TemplateIdentity]

    def predict(self, image: np.ndarray) -> DetectionResult: ...


class IDCardPipeline:
    def __init__(self, settings: Settings):
        # Validate local paths before YOLO can attempt an automatic download.
        for path in (settings.detector_path, settings.face_model_path):
            if not path.is_file():
                raise FileNotFoundError(f'Model file not found: {path}')
        if not settings.template_dir.is_dir():
            raise FileNotFoundError(f'Template directory not found: {settings.template_dir}')
        # Capture directory order once, associating identities with the exact
        # arrays used for matching. Filenames, not list indices, are API IDs.
        paths = [settings.template_dir / name for name in os.listdir(settings.template_dir)]
        paths = [path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        if not paths:
            raise ValueError('At least one pre-masked reference template is required')
        self.templates = [TemplateIdentity(path.name, path.stem) for path in paths]
        self._images = [load_template(path) for path in paths]

        from ultralytics import YOLO

        self._detector = YOLO(str(settings.detector_path)).eval()
        self._face_model = YOLO(str(settings.face_model_path)).eval()
        # The supplied face checkpoint is a pose model; its boxes are used and
        # its keypoints are intentionally ignored by remove_face().
        if self._detector.task != 'obb' or self._face_model.task not in {'detect', 'pose'}:
            raise ValueError('Expected an OBB card model and a face model that returns boxes')
        # Preserve model defaults apart from the explicit serving device.
        for model in (self._detector, self._face_model):
            model.overrides['device'] = settings.device
        self._threshold = settings.match_threshold
        self._lock = Lock()

    def predict(self, image: np.ndarray) -> DetectionResult:
        with self._lock:
            return analyze_card(
                self._detector, image, self._images, self._face_model, self._threshold,
            )
