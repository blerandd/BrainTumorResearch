
import numpy as np
from scipy import ndimage


def _validate_masks(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Mask shapes do not match: "
            f"{y_true.shape} and {y_pred.shape}"
        )

    if y_true.ndim != 2:
        raise ValueError("Boundary metrics require 2D masks.")

    return y_true, y_pred


def extract_boundary(mask):
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 2:
        raise ValueError("Expected a 2D binary mask.")

    structure = np.ones((3, 3), dtype=bool)

    eroded = ndimage.binary_erosion(
        mask,
        structure=structure,
        border_value=0
    )

    return mask & ~eroded


def calculate_boundary_distances(y_true, y_pred):
    y_true, y_pred = _validate_masks(y_true, y_pred)

    true_boundary = extract_boundary(y_true)
    pred_boundary = extract_boundary(y_pred)

    if not true_boundary.any() or not pred_boundary.any():
        return None, None

    distance_to_true = ndimage.distance_transform_edt(
        ~true_boundary
    )

    distance_to_pred = ndimage.distance_transform_edt(
        ~pred_boundary
    )

    pred_to_true = distance_to_true[pred_boundary]
    true_to_pred = distance_to_pred[true_boundary]

    return pred_to_true, true_to_pred


def calculate_hd95(y_true, y_pred):
    y_true, y_pred = _validate_masks(y_true, y_pred)

    true_boundary = extract_boundary(y_true)
    pred_boundary = extract_boundary(y_pred)

    if not true_boundary.any() and not pred_boundary.any():
        return 0.0

    if not true_boundary.any() or not pred_boundary.any():
        return float("inf")

    pred_to_true, true_to_pred = calculate_boundary_distances(
        y_true,
        y_pred
    )

    symmetric_distances = np.concatenate(
        [pred_to_true, true_to_pred]
    )

    return float(np.percentile(symmetric_distances, 95))


def calculate_assd(y_true, y_pred):
    y_true, y_pred = _validate_masks(y_true, y_pred)

    true_boundary = extract_boundary(y_true)
    pred_boundary = extract_boundary(y_pred)

    if not true_boundary.any() and not pred_boundary.any():
        return 0.0

    if not true_boundary.any() or not pred_boundary.any():
        return float("inf")

    pred_to_true, true_to_pred = calculate_boundary_distances(
        y_true,
        y_pred
    )

    symmetric_distances = np.concatenate(
        [pred_to_true, true_to_pred]
    )

    return float(np.mean(symmetric_distances))


def calculate_boundary_dice(
    y_true,
    y_pred,
    tolerance=2.0
):
    y_true, y_pred = _validate_masks(y_true, y_pred)

    if tolerance < 0:
        raise ValueError("Tolerance must be non-negative.")

    true_boundary = extract_boundary(y_true)
    pred_boundary = extract_boundary(y_pred)

    true_count = int(true_boundary.sum())
    pred_count = int(pred_boundary.sum())

    if true_count == 0 and pred_count == 0:
        return 1.0

    if true_count == 0 or pred_count == 0:
        return 0.0

    distance_to_true = ndimage.distance_transform_edt(
        ~true_boundary
    )

    distance_to_pred = ndimage.distance_transform_edt(
        ~pred_boundary
    )

    matched_predicted = int(
        np.sum(distance_to_true[pred_boundary] <= tolerance)
    )

    matched_true = int(
        np.sum(distance_to_pred[true_boundary] <= tolerance)
    )

    return float(
        (matched_predicted + matched_true) /
        (pred_count + true_count)
    )


def calculate_all_boundary_metrics(
    y_true,
    y_pred,
    tolerance=2.0
):
    return {
        "hd95_pixels": calculate_hd95(y_true, y_pred),
        "assd_pixels": calculate_assd(y_true, y_pred),
        "boundary_dice": calculate_boundary_dice(
            y_true,
            y_pred,
            tolerance=tolerance
        )
    }
