
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    roc_auc_score,
    ConfusionMatrixDisplay
)

from sklearn.preprocessing import label_binarize

from src.data.dataset_loader import BrainTumorDatasetLoader


ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "classification"
    / "custom_cnn_best.keras"
)

RESULTS_DIR = ROOT_DIR / "results" / "classification"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "pituitary"
]

NUM_CLASSES = len(CLASS_NAMES)


def load_trained_model():
    print("\nLOADING TRAINED CNN")
    print("=" * 55)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    if model.output_shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} output classes, "
            f"but model has {model.output_shape[-1]}."
        )

    print(f"Model: {MODEL_PATH}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")

    return model


def load_test_data():
    print("\nLOADING INDEPENDENT TEST DATA")
    print("=" * 55)

    loader = BrainTumorDatasetLoader(
        processed_dir=str(ROOT_DIR / "data" / "processed"),
        image_size=(256, 256),
        batch_size=16,
        seed=42
    )

    test_records = loader.metadata[
        loader.metadata["split"] == "test"
    ].copy()

    test_records = test_records.reset_index(drop=True)

    test_dataset = loader._build_classification_dataset("test")

    if test_records.empty:
        raise ValueError("The test partition is empty.")

    if not test_records["label"].isin(CLASS_NAMES).all():
        raise ValueError("Unexpected class found in test metadata.")

    print(f"Test images: {len(test_records)}")
    print(f"Test patients: {test_records['patient_id'].nunique()}")

    print("\nTEST CLASS DISTRIBUTION")

    print(
        test_records["label"].value_counts().reindex(
            CLASS_NAMES,
            fill_value=0
        )
    )

    return test_dataset, test_records


def generate_predictions(model, test_dataset):
    print("\nGENERATING TEST PREDICTIONS")
    print("=" * 55)

    y_true = np.concatenate(
        [
            labels.numpy()
            for _, labels in test_dataset
        ]
    ).astype(np.int32)

    probabilities = model.predict(
        test_dataset,
        verbose=1
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Prediction count does not match the test labels."
        )

    if probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(
            "Prediction probability dimensions do not match "
            "the number of classes."
        )

    print(f"Test samples: {len(y_true)}")
    print("Predictions generated successfully.")

    return y_true, y_pred, probabilities


def calculate_metrics(y_true, y_pred, probabilities):
    print("\nCLASSIFICATION METRICS")
    print("=" * 55)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Macro Precision:    {precision_macro:.4f}")
    print(f"Macro Recall:       {recall_macro:.4f}")
    print(f"Macro F1-score:     {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall:    {recall_weighted:.4f}")
    print(f"Weighted F1-score:  {f1_weighted:.4f}")

    print("\nPER-CLASS CLASSIFICATION REPORT")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    metrics = {
        "test_accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted)
    }

    try:
        macro_auc = roc_auc_score(
            y_true,
            probabilities,
            labels=list(range(NUM_CLASSES)),
            multi_class="ovr",
            average="macro"
        )

        weighted_auc = roc_auc_score(
            y_true,
            probabilities,
            labels=list(range(NUM_CLASSES)),
            multi_class="ovr",
            average="weighted"
        )

        metrics["macro_roc_auc"] = float(macro_auc)
        metrics["weighted_roc_auc"] = float(weighted_auc)

        print(f"Macro ROC-AUC:      {macro_auc:.4f}")
        print(f"Weighted ROC-AUC:   {weighted_auc:.4f}")

    except ValueError as error:
        print(f"ROC-AUC unavailable: {error}")

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        RESULTS_DIR / "classification_report.csv"
    )

    return metrics


def save_confusion_matrix(y_true, y_pred):
    print("\nGENERATING CONFUSION MATRIX")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    figure, axis = plt.subplots(figsize=(8, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    axis.set_title(
        "Custom CNN - Independent Test Confusion Matrix"
    )

    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("Actual Class")

    figure.tight_layout()

    figure.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print("Confusion matrix saved.")


def save_roc_curves(y_true, probabilities):
    print("\nGENERATING ROC CURVES")

    y_true_binary = label_binarize(
        y_true,
        classes=list(range(NUM_CLASSES))
    )

    figure, axis = plt.subplots(figsize=(9, 7))

    class_aucs = []

    for index, class_name in enumerate(CLASS_NAMES):
        if len(np.unique(y_true_binary[:, index])) < 2:
            print(
                f"Skipping {class_name}: "
                "both positive and negative samples are required."
            )
            continue

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true_binary[:, index],
            probabilities[:, index]
        )

        class_auc = auc(
            false_positive_rate,
            true_positive_rate
        )

        class_aucs.append(class_auc)

        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{class_name} (AUC = {class_auc:.4f})"
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    axis.set_title(
        "Custom CNN - One-vs-Rest ROC Curves"
    )

    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")

    axis.legend(loc="lower right")
    axis.grid(True)

    figure.tight_layout()

    figure.savefig(
        RESULTS_DIR / "roc_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"ROC curves saved. Classes evaluated: {len(class_aucs)}")


def save_predictions(y_true, y_pred, probabilities, test_records):
    print("\nSAVING TEST PREDICTIONS")

    if len(test_records) != len(y_true):
        raise ValueError(
            "Test metadata and prediction counts do not match."
        )

    predictions = pd.DataFrame({
        "sample_id": test_records["sample_id"].to_numpy(),
        "patient_id": test_records["patient_id"].to_numpy(),
        "actual_label": [
            CLASS_NAMES[index] for index in y_true
        ],
        "predicted_label": [
            CLASS_NAMES[index] for index in y_pred
        ],
        "correct_prediction": y_true == y_pred
    })

    for index, class_name in enumerate(CLASS_NAMES):
        predictions[f"probability_{class_name}"] = (
            probabilities[:, index]
        )

    predictions.to_csv(
        RESULTS_DIR / "test_predictions.csv",
        index=False
    )

    print("Test predictions saved.")


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nBRAIN TUMOR MRI CLASSIFICATION EVALUATION")
    print("=" * 60)

    model = load_trained_model()

    test_dataset, test_records = load_test_data()

    y_true, y_pred, probabilities = generate_predictions(
        model,
        test_dataset
    )

    print("\nEVALUATING MODEL ON TEST DATA")
    print("=" * 55)

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1
    )

    print(f"\nTest loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    metrics = calculate_metrics(
        y_true,
        y_pred,
        probabilities
    )

    metrics["test_loss"] = float(test_loss)
    metrics["test_accuracy"] = float(test_accuracy)
    metrics["test_samples"] = int(len(y_true))
    metrics["test_patients"] = int(
        test_records["patient_id"].nunique()
    )

    save_confusion_matrix(
        y_true,
        y_pred
    )

    save_roc_curves(
        y_true,
        probabilities
    )

    save_predictions(
        y_true,
        y_pred,
        probabilities,
        test_records
    )

    with open(
        RESULTS_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\nEVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Model: {MODEL_PATH}")
    print(f"Results: {RESULTS_DIR}")

    print("\nGENERATED FILES")
    print("- confusion_matrix.png")
    print("- roc_curves.png")
    print("- classification_report.csv")
    print("- metrics.json")
    print("- test_predictions.csv")


if __name__ == "__main__":
    main()
