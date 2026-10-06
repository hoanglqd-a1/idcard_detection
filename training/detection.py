"""Detect cards with YOLO OBB and classify them by face-masked template matching.

Images loaded by this module use RGB channel order. Labels are template indices.
"""

import os
import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Sequence

import cv2
import numpy as np

if __package__:
    from .utils.processing import (
        apply_mask, auto_canny, convert_rec2corners, crop_image, document_detect,
        draw_lines, expand_corners, extract_card, find_intersections, get_lines,
        load_image,
        crop_image_with_matrix, document_corners, four_point_transform_with_matrix,
        map_points, resize_transform,
    )
else:  # Preserve direct execution: python training/detection.py.
    from utils.processing import (
        apply_mask, auto_canny, convert_rec2corners, crop_image, document_detect,
        draw_lines, expand_corners, extract_card, find_intersections, get_lines,
        load_image,
        crop_image_with_matrix, document_corners, four_point_transform_with_matrix,
        map_points, resize_transform,
    )

if TYPE_CHECKING:
    from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
REFINEMENT_SIZE = (600, 400)  # OpenCV sizes are (width, height).
CARD_SIZE = (600, 400)
DEFAULT_MATCH_THRESHOLD = 0.8
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


class DetectionStatus(str, Enum):
    NO_CARD = 'no_card_detected'
    EXTRACTION_FAILED = 'extraction_failed'
    UNMATCHED = 'no_template_match'
    MATCHED = 'matched'


@dataclass(frozen=True)
class DetectionResult:
    """Inference data, independent of HTTP/image serialization.

    Corners are the detector's unexpanded OBB in input-image pixel coordinates,
    not the refined crop's boundary. Card is the unmasked RGB extraction.
    refined_corners contains the refined boundary in input-image coordinates
    after successful extraction; it is None when extraction fails.
    match_score is normalized correlation (-1..1), not a probability; it is
    retained even below threshold. None means matching was not performed or
    no templates were available. template_index uses the supplied list order.
    """

    status: DetectionStatus
    card: np.ndarray | None = None
    corners: np.ndarray | None = None
    detection_confidence: float | None = None
    template_index: int | None = None
    match_score: float | None = None
    failure_stage: Literal['crop', 'refinement'] | None = None
    processing_time_ms: float = 0.0
    refined_corners: np.ndarray | None = None

    @property
    def card_detected(self) -> bool:
        return self.corners is not None

    @property
    def is_supported(self) -> bool | None:
        """Template acceptance only; None when classification was not reached."""
        if self.status in (DetectionStatus.MATCHED, DetectionStatus.UNMATCHED):
            return self.status == DetectionStatus.MATCHED
        return None


def _first_corners(results):
    """Return the first detected OBB's corners, or None when no card is found."""
    if not results or results[0].obb is None or len(results[0].obb) == 0:
        return None
    return results[0].obb.xyxyxyxy[0].cpu().numpy()


def predict_corners(model: 'YOLO', image_dir):
    """Predict corners using the original CPU/320px inference settings."""
    return _first_corners(model.predict(image_dir, imgsz=320, conf=0.5, device='cpu'))


def detect_boundingbox(image, model: 'YOLO'):
    """Predict corners using the model's default inference settings."""
    return _first_corners(model(image))


def remove_face(model: 'YOLO', card: np.ndarray):
    """Black out face boxes, expanded by 30%, before template matching."""
    results = model.predict(card, imgsz=640, conf=0.5, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return card

    mask = np.full(card.shape[:2], 255, dtype=np.uint8)
    for box in results[0].boxes.xyxy:
        corners = convert_rec2corners(box.cpu().numpy())
        corners = expand_corners(card.shape, corners, expand_rate=0.3)
        top_left = tuple(corners[0].astype(int))
        bottom_right = tuple(corners[2].astype(int))
        cv2.rectangle(mask, top_left, bottom_right, 0, -1)
    return apply_mask(card, mask)


def load_templates(
    template_dir: str | Path, card_size: tuple[int, int] = CARD_SIZE,
) -> list[np.ndarray]:
    """Load pre-masked templates as RGB and resize for comparison.

    Template faces must already be blacked out; no face model is run here.
    Preserve directory listing order to retain the existing label mapping.
    """
    paths = (Path(template_dir) / name for name in os.listdir(template_dir))
    return [
        load_image(path, card_size)
        for path in paths
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def match(card, templates):
    """Return one normalized correlation score per template."""
    return np.array([
        cv2.matchTemplate(card, template, cv2.TM_CCOEFF_NORMED)[0, 0]
        for template in templates
    ])


def classify(card, templates, threshold=DEFAULT_MATCH_THRESHOLD):
    """Return the best template index, or None if no score exceeds threshold."""
    return classify_with_score(card, templates, threshold)[0]


def classify_with_score(
    card: np.ndarray, templates: Sequence[np.ndarray],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> tuple[int | None, float | None]:
    """Return accepted index and best correlation, including rejected scores."""
    scores = match(card, templates)
    if scores.size == 0:
        return None, None
    label = int(np.argmax(scores))
    score = float(scores[label])
    return (label if score > threshold else None), score


def detect_card(detect_model, image, templates, face_model, threshold=DEFAULT_MATCH_THRESHOLD):
    """Return (RGB card, template index), or (None, None) if cropping fails.

    A successfully cropped card with no template match returns (card, None).
    Model and configuration errors propagate to the caller.
    """
    result = analyze_card(detect_model, image, templates, face_model, threshold)
    return result.card, result.template_index


def analyze_card(
    detect_model: 'YOLO', image: np.ndarray, templates: Sequence[np.ndarray],
    face_model: 'YOLO', threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> DetectionResult:
    """Detect, refine and match a card from a full-resolution RGB image.

    Templates use 600x400 pixels. Border refinement uses a 600x400 temporary
    crop, but final pixels come directly from the input image.
    Retains model defaults, first-OBB selection and matching threshold.
    Model/configuration errors propagate instead of becoming normal outcomes.
    """
    if any(
        template.shape != (CARD_SIZE[1], CARD_SIZE[0], 3)
        for template in templates
    ):
        raise ValueError('Expected RGB templates sized 600x400 (width x height).')
    started = perf_counter()
    results = detect_model(image)
    corners = _first_corners(results)
    confidence = None
    if corners is not None:
        confidence = float(results[0].obb.conf[0].item())

    def finish(status: DetectionStatus, **kwargs) -> DetectionResult:
        return DetectionResult(
            status=status, corners=corners, detection_confidence=confidence,
            processing_time_ms=(perf_counter() - started) * 1000, **kwargs,
        )

    if corners is None:
        return finish(DetectionStatus.NO_CARD)
    refined_corners = None
    try:
        cropped, original_to_crop = crop_image_with_matrix(image, corners)
    except ValueError:
        return finish(DetectionStatus.EXTRACTION_FAILED, failure_stage='crop')
    crop_size = (cropped.shape[1], cropped.shape[0])
    working = cv2.resize(cropped, REFINEMENT_SIZE)
    refined = document_corners(working)
    if refined is None:
        return finish(DetectionStatus.EXTRACTION_FAILED, failure_stage='refinement')
    try:
        original_to_working = resize_transform(crop_size, REFINEMENT_SIZE) @ original_to_crop
        refined_corners = map_points(refined, np.linalg.inv(original_to_working))
        card, _ = four_point_transform_with_matrix(image, refined_corners, CARD_SIZE)
    except (ValueError, np.linalg.LinAlgError):
        return finish(DetectionStatus.EXTRACTION_FAILED, failure_stage='refinement')
    masked_card = remove_face(face_model, card)
    label, score = classify_with_score(masked_card, templates, threshold)
    return finish(
        DetectionStatus.MATCHED if label is not None else DetectionStatus.UNMATCHED,
        card=card, template_index=label, match_score=score,
        refined_corners=refined_corners,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description='Detect and refine a card with existing model weights.')
    parser.add_argument('--image', type=Path, default=ROOT / 'test_images' / 'image553.png')
    parser.add_argument('--output', type=Path, default=ROOT / 'detected_card_v2.png')
    args = parser.parse_args(argv)
    from ultralytics import YOLO

    detect_model = YOLO(str(ROOT / 'model' / 'yolov8s-detect.pt')).eval()
    face_model = YOLO(str(ROOT / 'model' / 'yolov8n-face.pt')).eval()
    templates = load_templates(ROOT / 'template_samples')
    image = load_image(args.image)
    result = analyze_card(detect_model, image, templates, face_model)
    card = result.card
    print('Status:', result.status.value)
    print('Detected card label:', result.template_index)
    print('Template similarity:', result.match_score)
    if card is None:
        print('No card could be extracted.')
        return
    output_path = args.output
    if not cv2.imwrite(str(output_path), cv2.cvtColor(card, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Could not save the detected card to {output_path}')


if __name__ == '__main__':
    main()
