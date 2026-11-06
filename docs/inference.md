# Inference details

[Back to README](../README.md)

| Asset | Role |
| --- | --- |
| `training/model/yolov8s-detect.pt` | YOLO OBB card detector; class `card` |
| `training/model/yolov8n-face.pt` | Face pose model; only its face bounding boxes are used |
| `training/template_samples/Template 0.jpg` … `Template 9.jpg` | Pre-masked reference images |

The checkpoint filenames do not fully describe their model tasks. Startup checks
the actual loaded tasks and fails clearly for missing/incompatible assets or an
empty template directory. Model paths are checked before invoking YOLO, preventing
automatic substitution/download of missing weights.

The training CLI and backend share one original-image pipeline:

```text
Full-resolution RGB input -> first YOLO OBB -> expanded temporary crop
-> resize temporary crop to 600 x 400 -> edge/Hough border refinement
-> map refined corners back to the original image
-> warp original pixels directly to 600 x 400
-> face masking -> downsample to 300 x 200 -> normalized template correlation -> strict score > 0.8
```

Reference templates are normalized to 600 x 400, then downsampled once to
300 x 200 with INTER_AREA. The masked query uses the same downsampling.
The returned card stays 600 x 400. Sizes are width by
height. YOLO performs its own internal input preparation; the API sends the
full uploaded image into detection.

The input face boxes are expanded by 30% and blacked out for matching. Returned
card images retain the face. Reference images are loaded directly without running
the face model again. Template IDs are filenames and display names are file stems;
the service binds each identity to its array using one directory listing, retaining
existing tie-breaking order. There are no verified document-type names.

The API retains the existing RGB NumPy inputs even though [Ultralytics documents BGR for
NumPy sources](https://docs.ultralytics.com/modes/predict/). Correcting that convention
is deferred because it could change results.
Original-preview geometry accounts for EXIF orientation without changing the
pixels sent into inference. Coordinates describe the refined card boundary when available, or the initial
OBB when extraction fails. No fallback crop is substituted when refinement fails.

## Sample observations

The supplied samples produced these local CPU smoke-test results using original-image extraction:

| Sample | Similarity | Result at `score > 0.8` |
| --- | --- | --- |
| `image553.png` | 0.836657 | Matched Template 5 |
| `image6.png` | 0.867860 | Matched Template 0 |

These results use 300 x 200 matching. The 0.8 threshold is unchanged; validation
on incorrect/nonmatching cards is deferred.
These are two sample observations, not an accuracy evaluation or a latency
benchmark. Border refinement can still select an inner line and clip content;
the coordinate mapping preserves the selected boundary rather than correcting it.

Smoke-test parity does not establish whether the reference images' colors were prepared correctly. Compare against source images before changing color handling.
