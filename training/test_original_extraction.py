"""Geometry regressions for the original-image extraction pipeline."""

import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import detection
import test_detection
from utils import processing


class OriginalExtractionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(7).integers(0, 256, (600, 900, 3), dtype=np.uint8)
        self.detected = np.array([[130, 110], [770, 70], [820, 480], [90, 520]], dtype=np.float32)
        self.refined = np.array([[155, 130], [745, 95], [790, 455], [120, 490]], dtype=np.float32)
        self.face = Mock()
        self.face.predict.return_value = []

    def model(self):
        model = test_detection.DetectionTests().detected_model()
        model.return_value[0].obb.xyxyxyxy[0].cpu.return_value.numpy.return_value = self.detected
        return model

    def refinement_points(self):
        crop, homography = processing.crop_image_with_matrix(self.image, self.detected)
        resize = processing.resize_transform((crop.shape[1], crop.shape[0]), detection.REFINEMENT_SIZE)
        return processing.map_points(self.refined, resize @ homography)

    def test_expanded_crop_homography_includes_margin(self):
        crop, transform = processing.crop_image_with_matrix(self.image, self.detected)
        expanded = processing.order_points(processing.expand_corners(self.image.shape, self.detected))
        target = [[0, 0], [crop.shape[1] - 1, 0],
                  [crop.shape[1] - 1, crop.shape[0] - 1], [0, crop.shape[0] - 1]]
        np.testing.assert_allclose(processing.map_points(expanded, transform), target, atol=1e-3)
        self.assertFalse(np.allclose(expanded, self.detected))

    def test_resize_uses_pixel_centers(self):
        transform = processing.resize_transform((100, 80), (200, 40))
        np.testing.assert_allclose(processing.map_points([[0, 0], [10, 20]], transform),
                                   [[0.5, -0.25], [20.5, 9.75]])

    def test_original_pixels_and_refined_corners_are_preserved(self):
        # Expected image is a direct warp of the known quadrilateral, independent
        # of our crop/resize helpers. A second warp of the temporary crop differs.
        target = np.array([[0, 0], [599, 0], [599, 399], [0, 399]], dtype=np.float32)
        expected = cv2.warpPerspective(
            self.image, cv2.getPerspectiveTransform(self.refined, target), (600, 400),
        )
        original = self.image.copy()
        model = self.model()
        with patch.object(detection, 'document_corners', return_value=self.refinement_points()) as refine:
            result = detection.analyze_card(model, self.image, [cv2.resize(expected, detection.MATCH_SIZE, interpolation=cv2.INTER_AREA)], self.face)
        model.assert_called_once_with(self.image)
        self.assertEqual(refine.call_args.args[0].shape, (400, 600, 3))
        np.testing.assert_allclose(result.refined_corners, self.refined, atol=1e-4)
        np.testing.assert_array_equal(result.corners, self.detected)
        np.testing.assert_array_equal(result.card, expected)
        np.testing.assert_array_equal(self.image, original)
        self.assertEqual(result.card.shape, (400, 600, 3))
        self.assertEqual(result.status, detection.DetectionStatus.MATCHED)
        self.assertEqual(result.template_index, 0)
        self.assertAlmostEqual(result.match_score, 1, places=5)

    def test_matching_downsamples_after_masking_without_changing_display_card(self):
        masked = np.zeros((400, 600, 3), dtype=np.uint8)
        masked[75:275, 120:420] = 180
        with patch.object(detection, 'document_corners', return_value=self.refinement_points()), \
             patch.object(detection, 'remove_face', return_value=masked) as mask, \
             patch.object(detection, 'classify_with_score', return_value=(None, 0.7)) as classify:
            result = detection.analyze_card(self.model(), self.image, [], self.face)
        self.assertIs(mask.call_args.args[1], result.card)
        self.assertEqual(result.card.shape, (400, 600, 3))
        np.testing.assert_array_equal(classify.call_args.args[0],
                                      cv2.resize(masked, (300, 200), interpolation=cv2.INTER_AREA))

    def test_refinement_failure_does_not_fall_back_to_yolo_crop(self):
        with patch.object(detection, 'document_corners', return_value=None):
            result = detection.analyze_card(self.model(), self.image, [], self.face)
        self.assertTrue(result.card_detected)
        self.assertEqual(result.failure_stage, 'refinement')
        self.assertIsNone(result.card)
        self.assertIsNone(result.refined_corners)
        self.face.predict.assert_not_called()

    def test_invalid_geometry_is_reported_as_extraction_failure(self):
        for stage in ('crop', 'refinement'):
            with self.subTest(stage=stage):
                model = self.model()
                if stage == 'crop':
                    model.return_value[0].obb.xyxyxyxy[0].cpu.return_value.numpy.return_value = np.zeros((4, 2))
                with patch.object(detection, 'document_corners', return_value=np.zeros((4, 2))):
                    result = detection.analyze_card(model, self.image, [], self.face)
                self.assertEqual(result.failure_stage, stage)
                self.assertIsNone(result.card)

    def test_incorrect_template_size_is_a_configuration_error(self):
        with self.assertRaisesRegex(ValueError, '300x200'):
            detection.analyze_card(self.model(), self.image,
                                  [np.zeros((320, 640, 3), dtype=np.uint8)], self.face)

    def test_blank_image_has_no_refined_corners(self):
        self.assertIsNone(processing.document_corners(np.zeros((400, 600, 3), dtype=np.uint8)))

    def test_points_at_infinity_are_rejected(self):
        with self.assertRaises(ValueError):
            processing.map_points([[1, 2]], np.zeros((3, 3)))


if __name__ == '__main__':
    unittest.main()
