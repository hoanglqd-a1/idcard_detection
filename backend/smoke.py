"""Run real weights and check serving parity with the legacy entry point.

Run from the repository root: python -m backend.smoke
Outputs are local, ignored artifacts; no recognition accuracy is inferred.
"""

import json

import numpy as np
from PIL import Image

from backend.app.config import ROOT, Settings
from backend.app.inference.pipeline import IDCardPipeline
from backend.app.services.analysis import analyze_image, decode_image
from training.detection import detect_card


def main():
    settings = Settings()
    pipeline = IDCardPipeline(settings)
    output = ROOT / 'artifacts/smoke'
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted((ROOT / 'training/test_images').glob('*.png')):
        data = path.read_bytes()
        working, _, _, _ = decode_image(data, settings)
        legacy_card, legacy_label = detect_card(
            pipeline._detector, working, pipeline._images, pipeline._face_model,
            settings.match_threshold,
        )
        result = pipeline.predict(working)
        assert result.template_index == legacy_label
        if legacy_card is None:
            assert result.card is None
        else:
            np.testing.assert_array_equal(result.card, legacy_card)
            Image.fromarray(result.card).save(output / f'{path.stem}-crop.png')
        response = analyze_image(data, pipeline, settings)
        report = {'sample': path.name, 'legacy_parity': True,
                  **response.model_dump(mode='json', exclude={'extracted_card'})}
        reports.append(report)
        print(json.dumps(report))
    if not reports:
        raise FileNotFoundError('No PNG smoke-test images found')
    (output / 'results.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
