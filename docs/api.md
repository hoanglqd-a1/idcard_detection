# API reference

[Back to README](../README.md)

Interactive API documentation is available at http://localhost:8000/docs while the backend is running.

### `GET /api/v1/health`

Returns readiness and template count. The service fails startup if model loading
fails; an uninitialized service returns HTTP 503 rather than claiming readiness.

### `POST /api/v1/analyze`

Accepts `multipart/form-data` with an image in the `file` field. Actual image
decoding determines validity; a filename or declared MIME type is not trusted.

```sh
curl -F "file=@training/test_images/image6.png" http://localhost:8000/api/v1/analyze
```

On Windows use `curl.exe`. Normal CV outcomes return HTTP 200:

| `status` | Meaning |
| --- | --- |
| `matched` | Extracted card matched a reference above the threshold |
| `no_template_match` | Card extracted; no score exceeded the threshold |
| `no_card_detected` | Detector did not return an OBB |
| `extraction_failed` | Card detected; crop or border refinement failed |

The response includes `card_detected`, `template_id`, `template_name`,
`detection_confidence`, `match_score`, `match_threshold`, `is_supported`,
`corners`, `image_width`, `image_height`, `extracted_card`, `failure_stage`,
`processing_time_ms`, and `inference_time_ms`.

- `is_supported` is `null` when classification was not reached. It means a
  reference matched, not that a legal document type was verified.
- `match_score` is correlation in `[-1, 1]`, not a calibrated probability.
- `extracted_card` is a PNG data URL or `null`; the browser retains the original.
- Corners use pixel coordinates in the EXIF-oriented original preview.
- Processing time covers decoding, waiting for the shared model, inference,
  and crop encoding. It excludes upload transfer and response transmission.
- Inference time is the existing pipeline's wall time, excluding lock wait.

Errors use `{"error":{"code":"invalid_image","message":"..."}}`.
HTTP 413 means a byte/pixel limit was exceeded, 415 an unsupported image format,
422 missing/invalid input, 503 not ready, and 500 an unexpected inference error.
Responses are marked `Cache-Control: no-store`. Request bodies are bounded even
when sent without a Content-Length header.
