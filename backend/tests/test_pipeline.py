from unittest.mock import patch

import numpy as np
from PIL import Image

from backend.app.config import Settings
from backend.app.inference.pipeline import IDCardPipeline


def test_missing_assets_do_not_trigger_model_download(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        IDCardPipeline(Settings(detector_path=tmp_path / 'missing.pt'))


def test_template_identity_stays_associated_with_loaded_pixels(tmp_path):
    from unittest.mock import Mock
    for name, color in [('Template B.png', (30, 80, 120)), ('Template A.png', (90, 20, 10))]:
        Image.new('RGB', (64, 32), color).save(tmp_path / name)
    (tmp_path / 'notes.txt').write_text('skip')
    model = tmp_path / 'model.pt'
    model.touch()
    detector, face = Mock(task='obb'), Mock(task='pose')
    detector.overrides, face.overrides = {}, {}
    with patch('ultralytics.YOLO') as loader, patch(
        'backend.app.inference.pipeline.os.listdir',
        return_value=['Template B.png', 'notes.txt', 'Template A.png', 'model.pt'],
    ):
        loader.return_value.eval.side_effect = [detector, face]
        pipeline = IDCardPipeline(Settings(detector_path=model, face_model_path=model, template_dir=tmp_path))
    assert [template.id for template in pipeline.templates] == ['Template B.png', 'Template A.png']
    assert all(image.shape == (400, 600, 3) for image in pipeline._images)
    assert pipeline._images[0][0, 0].tolist() == [30, 80, 120]
    detector.predict.assert_not_called()
    face.predict.assert_not_called()
    with patch('backend.app.inference.pipeline.analyze_card') as analyze:
        image = np.zeros((900, 1200, 3), dtype=np.uint8)
        pipeline.predict(image)
        assert analyze.call_args.args[0] is detector
        assert analyze.call_args.args[1] is image
        assert analyze.call_args.args[2] is pipeline._images
        assert analyze.call_args.args[3] is face
        assert analyze.call_args.args[4] == 0.8
