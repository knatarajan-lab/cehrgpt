import numpy as np

from cehrgpt.analysis.privacy.attribute_inference import cal_f1_score


def test_precision_uses_the_matched_patient():
    real = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
    synthetic = np.array([[1, 0, 0, 0], [1, 1, 1, 1]])

    # Both real patients match the second synthetic patient, who has 4 attributes
    f1, precision, recall = cal_f1_score(real, synthetic, np.array([1, 1]))

    np.testing.assert_allclose(precision, [0.5, 0.5])
    np.testing.assert_allclose(recall, [1.0, 1.0])
    np.testing.assert_allclose(f1, [2 / 3, 2 / 3])


def test_scores_stay_within_bounds_when_match_order_differs():
    real = np.array([[1, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 0]])
    synthetic = np.array([[1, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 0]])

    f1, precision, recall = cal_f1_score(real, synthetic, np.array([1, 2, 1]))

    np.testing.assert_allclose(precision, [0.75, 1.0, 0.25])
    np.testing.assert_allclose(recall, [1.0, 0.5, 1.0])
    np.testing.assert_allclose(f1, [6 / 7, 2 / 3, 0.4])


def test_patient_without_sensitive_attributes_scores_zero():
    real = np.array([[0, 0], [1, 0]])
    synthetic = np.array([[0, 0], [1, 0]])

    f1, precision, recall = cal_f1_score(real, synthetic, np.array([1, 0]))

    np.testing.assert_allclose(precision, [0.0, 0.0])
    np.testing.assert_allclose(recall, [0.0, 0.0])
    np.testing.assert_allclose(f1, [0.0, 0.0])
