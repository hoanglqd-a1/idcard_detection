from io import BytesIO
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.inference.pipeline import TemplateIdentity
from backend.app.main import create_app
from training.detection import DetectionResult, DetectionStatus


def image_bytes(size=(1280, 640), format='PNG', orientation=None):
    output = BytesIO()
    image = Image.new('RGB', size, (50, 100, 150))
    if orientation:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(output, format=format, exif=exif)
    else:
        image.save(output, format=format)
    return output.getvalue()


@pytest.fixture
def api():
    pipeline = Mock()
    pipeline.templates = [TemplateIdentity('Template 3.jpg', 'Template 3')]
    pipeline.predict.return_value = DetectionResult(DetectionStatus.NO_CARD)
    factory = Mock(return_value=pipeline)
    with TestClient(create_app(Settings(), factory), raise_server_exceptions=False) as client:
        yield client, pipeline, factory


def test_startup_loads_once_and_health(api):
    client, pipeline, factory = api
    assert client.get('/api/v1/health').json() == {
        'status': 'ready', 'models_loaded': True, 'template_count': 1,
    }
    for _ in range(2):
        response = client.post('/api/v1/analyze', files={'file': ('photo.png', image_bytes(), 'image/png')})
        assert response.status_code == 200
        assert response.json()['status'] == 'no_card_detected'
        assert response.json()['is_supported'] is None
    factory.assert_called_once()
    assert pipeline.predict.call_count == 2
    assert pipeline.predict.call_args.args[0].shape == (320, 640, 3)


@pytest.mark.parametrize('status,has_crop,index', [
    (DetectionStatus.MATCHED, True, 0),
    (DetectionStatus.UNMATCHED, True, None),
    (DetectionStatus.EXTRACTION_FAILED, False, None),
])
def test_outcomes_geometry_and_rgb_crop(api, status, has_crop, index):
    client, pipeline, _ = api
    crop = np.full((320, 640, 3), (200, 50, 10), dtype=np.uint8) if has_crop else None
    pipeline.predict.return_value = DetectionResult(
        status=status, card=crop, corners=np.array([[10, 20], [600, 20], [600, 300], [10, 300]]),
        detection_confidence=0.95, template_index=index,
        match_score=0.9 if index is not None else 0.6 if has_crop else None,
        failure_stage=None if has_crop else 'refinement', processing_time_ms=1.0,
    )
    response = client.post('/api/v1/analyze', files={'file': ('photo.png', image_bytes())})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['status'] == status.value
    assert body['corners'][0] == {'x': 20, 'y': 40}
    assert body['card_detected'] is True
    assert body['template_id'] == ('Template 3.jpg' if index is not None else None)
    assert body['is_supported'] == (index is not None if has_crop else None)
    assert body['inference_time_ms'] == 1.0
    assert body['processing_time_ms'] >= 0
    assert response.headers['cache-control'] == 'no-store'
    if has_crop:
        import base64
        image = Image.open(BytesIO(base64.b64decode(body['extracted_card'].split(',')[1])))
        assert image.getpixel((0, 0)) == (200, 50, 10)
    else:
        assert body['extracted_card'] is None


def test_exif_overlay_keeps_legacy_inference_pixels(api):
    client, pipeline, _ = api
    pipeline.predict.return_value = DetectionResult(
        DetectionStatus.EXTRACTION_FAILED, corners=np.array([[10, 20], [600, 20], [600, 300], [10, 300]]),
    )
    response = client.post('/api/v1/analyze', files={'file': ('rotated.jpg', image_bytes(format='JPEG', orientation=6))})
    body = response.json()
    assert body['image_width'] == 640
    assert body['image_height'] == 1280
    assert body['corners'][0] == {'x': 599, 'y': 20}
    assert pipeline.predict.call_args.args[0].shape == (320, 640, 3)


@pytest.mark.parametrize('data,status', [(b'', 422), (b'not an image', 422), (image_bytes(format='GIF'), 415)])
def test_invalid_upload_never_runs_inference(api, data, status):
    client, pipeline, _ = api
    response = client.post('/api/v1/analyze', files={'file': ('fake.png', data, 'image/png')})
    assert response.status_code == status
    assert response.json()['error']['message']
    pipeline.predict.assert_not_called()


def test_missing_upload(api):
    response = api[0].post('/api/v1/analyze')
    assert response.status_code == 422
    assert response.json()['error']['code'] == 'invalid_request'


@pytest.mark.parametrize('settings,data', [
    (Settings(max_upload_bytes=100), image_bytes()),
    (Settings(max_image_pixels=100), image_bytes((20, 20))),
])
def test_limits(settings, data):
    pipeline = Mock(templates=[])
    with TestClient(create_app(settings, lambda _: pipeline)) as client:
        response = client.post('/api/v1/analyze', files={'file': ('image.png', data)})
    assert response.status_code == 413
    pipeline.predict.assert_not_called()


def test_chunked_request_limit():
    pipeline = Mock(templates=[])
    with TestClient(create_app(Settings(max_upload_bytes=100), lambda _: pipeline)) as client:
        response = client.post('/api/v1/analyze', content=iter([b'x' * 70_000]),
                               headers={'content-type': 'multipart/form-data; boundary=abc'})
    assert response.status_code == 413
    assert response.json()['error']['code'] == 'file_too_large'
    pipeline.predict.assert_not_called()


def test_internal_error_is_not_no_detection(api):
    client, pipeline, _ = api
    pipeline.predict.side_effect = RuntimeError('private model path')
    response = client.post('/api/v1/analyze', files={'file': ('photo.png', image_bytes())})
    assert response.status_code == 500
    assert response.json()['error']['code'] == 'inference_error'
    assert 'private model path' not in response.text


def test_not_ready(api):
    client, _, _ = api
    client.app.state.pipeline = None
    assert client.get('/api/v1/health').status_code == 503
    response = client.post('/api/v1/analyze', files={'file': ('photo.png', image_bytes())})
    assert response.status_code == 503


def test_startup_failure_is_visible():
    factory = Mock(side_effect=FileNotFoundError('missing weights'))
    with pytest.raises(FileNotFoundError, match='missing weights'):
        with TestClient(create_app(Settings(), factory)):
            pass
