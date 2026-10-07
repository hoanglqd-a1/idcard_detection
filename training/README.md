# ID Card Detection - Training Workspace

Prepare the MIDV500 dataset, train a YOLOv8 oriented bounding box (OBB) model, and try card extraction and template-based classification.

This folder contains model training and experimentation code. The repository's
[CardScope application](../README.md) adds a FastAPI backend and React frontend
around this inference pipeline; training remains a separate concern.

## Prepare the dataset

Inside the container, run:

```bash
python download_dataset.py
```

The default output is `/workspace/datasets/MIDV500`. To choose another location:

```bash
python download_dataset.py --output-dir /workspace/datasets/custom-midv500
```

Use a directory under `/workspace` for persistence, or add another bind mount to the Docker command for an external dataset drive. The existing `docker.sh` mounts only the project directory.

The generated layout is:

```text
datasets/MIDV500/
|-- data.yaml
|-- images/
|   |-- train/
|   `-- val/
|-- labels/
|   |-- train/
|   `-- val/
`-- temp/                 # Downloaded ZIPs and extracted source data
```

The script samples every seventh source frame, filters annotations, and creates an 80/20 split with a fixed random seed. Labels contain class `0` followed by four normalized corner coordinates. This is an OBB dataset with one class, `card`; document types are classified later by template matching.

Generated `data.yaml`:

```yaml
path: /workspace/datasets/MIDV500
train: images/train
val: images/val

names:
  0: card
```

The `path` is absolute and must be valid inside the environment running training. If the dataset was generated on Windows, update its YAML path for the container before training.

## Train the detector

Ensure the initial OBB checkpoint is available at:

```text
yolo_training_report/yolov8s-obb.pt
```

Then run inside the container:

```bash
python train.py
# For a custom dataset:
python train.py --data /workspace/datasets/custom-midv500/data.yaml
```

Outputs:

- `yolo_training_report/`: training configuration, metrics, plots, and checkpoints.
- `yolo_training_report/weights/best.pt`: best training checkpoint.
- `model/yolov8s-detect.pt`: copy of that checkpoint for inference.

Training reuses `yolo_training_report/` and replaces the exported model. Preserve previous reports or weights separately if you need to compare runs. Despite the exported filename, this remains an **OBB** model.

## Try detection and classification

Ensure these assets are available:

- `model/yolov8s-detect.pt`: trained card detector.
- `model/yolov8n-face.pt`: separately supplied face detector; this repository does not train it.
- Reference card images with faces already blacked out in `template_samples/`.
- The selected input image in `test_images/`.

Run:

```bash
python detection.py
```

The CLI reads `test_images/image553.png` at its original resolution. Successful
extraction writes `detected_card_v2.png` and prints the status, template index,
and similarity. Choose an image or destination:

```bash
python detection.py --image test_images/image6.png --output card.png
```

If extraction fails, no new image is written; an older output file may still exist.

The pipeline detects an oriented card box, crops it, refines its border, masks faces for comparison, and selects the template with the highest normalized correlation above the threshold. Helpers use RGB images and `(width, height)` resize dimensions.

`detect_card(...)` returns:

- `(card, label)` for a successful extraction and template match.
- `(card, None)` for an extracted card without a sufficient template match.
- `(None, None)` when a card cannot be detected or cropped.

Model and configuration errors propagate to the caller. The returned card retains its face; masking is applied to the comparison image.

Labels are indices in the loaded template list, not document names. Loading preserves directory listing order, which is not a stable product-level class mapping. A backend should introduce an explicit template-to-document-type mapping before exposing these labels to users.

### Structured inference results

Use `analyze_card(...)` for scores, geometry, and failure details. It accepts the
same already-loaded models, RGB image, templates, and threshold as `detect_card`.
It does not load models or resize the input on entry. From the repository root:

```python
from ultralytics import YOLO
from training.detection import ROOT, analyze_card, load_image, load_templates

# Initialize once and reuse for subsequent images.
detector = YOLO(str(ROOT / 'model' / 'yolov8s-detect.pt')).eval()
face_detector = YOLO(str(ROOT / 'model' / 'yolov8n-face.pt')).eval()
templates = load_templates(ROOT / 'template_samples')

image = load_image(ROOT / 'test_images' / 'image6.png')
result = analyze_card(detector, image, templates, face_detector)
print(result.status.value, result.template_index, result.match_score)
```

The returned `DetectionResult` contains:

| Field | Meaning |
| --- | --- |
| `status` | `matched`, `no_template_match`, `no_card_detected`, or `extraction_failed` |
| `card_detected` | Whether the detector returned corners, even if extraction failed |
| `card` | Unmasked rectified RGB NumPy array, or `None` |
| `corners` | First detected OBB's four points in the supplied image's pixel coordinates |
| `refined_corners` | Refined boundary in input-image coordinates after successful extraction; otherwise `None` |
| `detection_confidence` | YOLO confidence for that OBB, or `None` |
| `template_index` | Accepted template's list index, or `None` |
| `match_score` | Best normalized correlation, retained for rejected matches; `None` if unavailable |
| `is_supported` | Whether a reference matched; `None` when classification was not reached |
| `failure_stage` | `crop` or `refinement` for extraction failures; otherwise `None` |
| `processing_time_ms` | Pipeline wall time, excluding model/template loading and input decoding |

Template similarity is not a calibrated classification probability. An unmatched
result means **no reference template matched**, not proof of an unsupported
document type. Empty template lists also produce an unmatched result, with no
score. The strict default acceptance rule is `match_score > 0.8`.

Corners describe the initial detection, not the refined extraction boundary.
If the caller resizes an image, it must map coordinates back for an original-image
overlay. Image encoding and API serialization belong to the caller; this result
contains NumPy arrays and is not directly JSON serializable. Unexpected model or
configuration errors still propagate instead of being labeled as no detection.

`detect_card(...)` remains a compatibility wrapper returning `(card, label)`.
All matching entry points default to `DEFAULT_MATCH_THRESHOLD = 0.8`.
`classify(...)` retains its index-only return;
`classify_with_score(...)` exposes the accepted index and best score together.

Templates are loaded as RGB, normalized to `CARD_SIZE`, then downsampled to
`MATCH_SIZE` with OpenCV INTER_AREA once at loading. Loading does not run the
face detector. `load_templates(template_dir, card_size=MATCH_SIZE)` no longer
takes a face model. The face model is still required to mask the extracted input
card before comparison. Use the same black-mask convention for reference images.
When saving RGB arrays with OpenCV, convert them with
`cv2.cvtColor(image, cv2.COLOR_RGB2BGR)` before `cv2.imwrite` to preserve colors.

### Original-image extraction

The shared training and backend pipeline preserves border refinement and samples the final card directly from
the original image:

```text
Full-resolution RGB input → YOLO OBB → expanded temporary perspective crop
-> resize temporary crop to 600 x 400 -> locate refined border corners
→ invert resize and crop transforms → refined corners in original image
→ warp original image directly to 600 × 400 → face mask → template matching
```

Sizes are `(width, height)`: the final RGB array has shape `(400, 600, 3)`.
`REFINEMENT_SIZE = (600, 400)` sets the edge/Hough working scale;
`CARD_SIZE = (600, 400)` controls extraction and reference normalization.
`MATCH_SIZE = (300, 200)` controls matching only. The extracted card is face-masked
before downsampling; the returned display card remains unmasked at 600 x 400.
The 0.8 threshold is unchanged; validation on incorrect/nonmatching cards remains future work.
The detector still performs its own internal input preparation. Its settings,
first-OBB choice, RGB convention, 5% crop expansion, and strict `0.8` matching
threshold are unchanged. No models or template image files are rewritten.

```python
from training.detection import MATCH_SIZE, analyze_card, load_templates, load_image

templates = load_templates(ROOT / 'template_samples', card_size=MATCH_SIZE)
image = load_image(ROOT / 'test_images/image6.png')  # No whole-photo resize.
result = analyze_card(detector, image, templates, face_detector)
```

The crop helper exposes the homography for the **expanded** detector region.
The refinement-to-original mapping inverts `resize_matrix @ crop_homography`,
including OpenCV resize's pixel-center offsets. The final warp uses the original
pixels, not the temporary refinement image. Failed refinement still returns
`extraction_failed`; there is no automatic fallback to a YOLO-only crop.

Run `python -m backend.smoke` from the repository root to check both supplied
images with real weights and write crops/results to `artifacts/smoke/`.
The previous two-mode comparison script has been removed.

Border selection still uses the existing Hough-line grouping and intersection
rules. It can select an inner edge and clip card content; original-resolution
sampling improves detail but does not fix incorrect border selection.

## Tests

With dependencies installed, run:

```bash
python -m unittest test_download_dataset test_detection test_original_extraction -v
```

These tests use temporary data and mocked model/FTP calls. They do not download MIDV500, train a model, or measure real-world recognition accuracy.

## Product handoff and validation

The training workspace produces `model/yolov8s-detect.pt` for the inference service.
The application reuses `analyze_card(...)`, exposes filename-based template
identities, and displays detection and extraction results. Run `python -m
backend.smoke` from the repository root to check real-model parity with the
training entry point. See the [application README](../README.md) for API, frontend,
and container instructions.
