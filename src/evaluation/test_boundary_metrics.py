
import numpy as np

from src.evaluation.boundary_metrics import (
    calculate_hd95,
    calculate_assd,
    calculate_boundary_dice,
    calculate_all_boundary_metrics
)


def run_tests():
    print("\nBOUNDARY METRICS UNIT TESTS")
    print("=" * 60)

    true_mask = np.zeros((32, 32), dtype=bool)
    true_mask[10, 10] = True

    predicted_mask = true_mask.copy()

    print("\nTEST 1: IDENTICAL MASKS")

    hd95 = calculate_hd95(true_mask, predicted_mask)
    assd = calculate_assd(true_mask, predicted_mask)
    boundary_dice = calculate_boundary_dice(
        true_mask,
        predicted_mask,
        tolerance=2
    )

    assert hd95 == 0.0
    assert assd == 0.0
    assert boundary_dice == 1.0

    print("HD95:", hd95)
    print("ASSD:", assd)
    print("Boundary Dice:", boundary_dice)
    print("PASSED")

    print("\nTEST 2: MASKS SEPARATED BY THREE PIXELS")

    predicted_mask = np.zeros((32, 32), dtype=bool)
    predicted_mask[10, 13] = True

    hd95 = calculate_hd95(true_mask, predicted_mask)
    assd = calculate_assd(true_mask, predicted_mask)

    assert np.isclose(hd95, 3.0)
    assert np.isclose(assd, 3.0)

    boundary_dice_2 = calculate_boundary_dice(
        true_mask,
        predicted_mask,
        tolerance=2
    )

    boundary_dice_3 = calculate_boundary_dice(
        true_mask,
        predicted_mask,
        tolerance=3
    )

    assert boundary_dice_2 == 0.0
    assert boundary_dice_3 == 1.0

    print("HD95:", hd95)
    print("ASSD:", assd)
    print("Boundary Dice at 2 pixels:", boundary_dice_2)
    print("Boundary Dice at 3 pixels:", boundary_dice_3)
    print("PASSED")

    print("\nTEST 3: EMPTY MASKS")

    empty_mask = np.zeros((32, 32), dtype=bool)

    assert calculate_hd95(empty_mask, empty_mask) == 0.0
    assert calculate_assd(empty_mask, empty_mask) == 0.0
    assert calculate_boundary_dice(empty_mask, empty_mask) == 1.0

    print("Both masks empty: PASSED")

    print("\nTEST 4: ONE EMPTY MASK")

    assert np.isinf(calculate_hd95(true_mask, empty_mask))
    assert np.isinf(calculate_assd(true_mask, empty_mask))
    assert calculate_boundary_dice(true_mask, empty_mask) == 0.0

    print("One mask empty: PASSED")

    print("\nTEST 5: COMPLETE METRIC FUNCTION")

    result = calculate_all_boundary_metrics(
        true_mask,
        true_mask,
        tolerance=2
    )

    print(result)

    assert result["hd95_pixels"] == 0.0
    assert result["assd_pixels"] == 0.0
    assert result["boundary_dice"] == 1.0

    print("PASSED")

    print("\nALL BOUNDARY METRIC TESTS PASSED")


if __name__ == "__main__":
    run_tests()
