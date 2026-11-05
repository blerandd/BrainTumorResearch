
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

MODEL_PATH = ROOT_DIR / "models" / "custom_brain_tumor_cnn.keras"

RESULTS_DIR = ROOT_DIR / "results" / "classification"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "pituitary"
]

NUM_CLASSES = len(CLASS_NAMES)


def load_trained_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Save your trained model to this location first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nMODEL LOADED SUCCESSFULLY")
    print("=" * 55)
    print(f"Model: {MODEL_PATH}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    return model


def load_test_data():
    print("\nLOADING TEST DATA")
    print("=" * 55)

    loader = BrainTumorDatasetLoader()

    datasets = loader.get_classification_datasets()

    test_dataset = datasets["test"]

    print(f"Test images: {len(loader.metadata[loader.metadata['split'] == 'test'])}")

    return test_dataset


def get_predictions(model, test_dataset):
    print("\nGENERATING TEST PREDICTIONS")
    print("=" * 55)

    y_true = []

    for _, labels in test_dataset:
        y_true.extend(labels.numpy())

    y_true = np.asarray(y_true, dtype=np.int32)

    probabilities = model.predict(
        test_dataset,
        verbose=1
    )

    y_pred = np.argmax(probabilities, axis=1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Number of predictions does not match number of labels."
        )

    print(f"Number of test samples: {len(y_true)}")

    return y_true, y_pred, probabilities


def calculate_metrics(y_true, y_pred, probabilities):
    print("\nCLASSIFICATION RESULTS")
    print("=" * 55)

    accuracy = accuracy_score(y_true, y_pred)

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

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Macro Precision:    {precision_macro:.4f}")
    print(f"Macro Recall:       {recall_macro:.4f}")
    print(f"Macro F1-score:     {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall:    {recall_weighted:.4f}")
    print(f"Weighted F1-score:  {f1_weighted:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

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
        "accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted)
    }

    try:
        y_true_binary = label_binarize(
            y_true,
            classes=list(range(NUM_CLASSES))
        )

        macro_auc = roc_auc_score(
            y_true_binary,
            probabilities,
            average="macro",
            multi_class="ovr"
        )

        weighted_auc = roc_auc_score(
            y_true_binary,
            probabilities,
            average="weighted",
            multi_class="ovr"
        )

        metrics["macro_roc_auc"] = float(macro_auc)
        metrics["weighted_roc_auc"] = float(weighted_auc)

        print(f"Macro ROC-AUC:      {macro_auc:.4f}")
        print(f"Weighted ROC-AUC:   {weighted_auc:.4f}")

    except ValueError as error:
        print(f"ROC-AUC could not be calculated: {error}")

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        RESULTS_DIR / "classification_report.csv"
    )

    with open(
        RESULTS_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    return metrics


def save_confusion_matrix(y_true, y_pred):
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

    axis.set_title("Custom CNN - Test Confusion Matrix")
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print("\nConfusion matrix saved.")


def save_roc_curves(y_true, probabilities):
    y_true_binary = label_binarize(
        y_true,
        classes=list(range(NUM_CLASSES))
    )

    figure, axis = plt.subplots(figsize=(9, 7))

    valid_classes = 0

    for index, class_name in enumerate(CLASS_NAMES):
        if len(np.unique(y_true_binary[:, index])) < 2:
            print(f"Skipping ROC curve for {class_name}: missing class samples.")
            continue

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true_binary[:, index],
            probabilities[:, index]
        )

        class_auc = auc(
            false_positive_rate,
            true_positive_rate
        )

        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{class_name} (AUC = {class_auc:.4f})"
        )

        valid_classes += 1

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    axis.set_title("Custom CNN - One-vs-Rest ROC Curves")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    axis.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "roc_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"ROC curves saved. Valid classes: {valid_classes}")


def save_predictions(y_true, y_pred, probabilities):
    data = {
        "actual_label": [
            CLASS_NAMES[index] for index in y_true
        ],
        "predicted_label": [
            CLASS_NAMES[index] for index in y_pred
        ],
        "correct_prediction": y_true == y_pred
    }

    for index, class_name in enumerate(CLASS_NAMES):
        data[f"probability_{class_name}"] = probabilities[:, index]

    predictions_df = pd.DataFrame(data)

    predictions_df.to_csv(
        RESULTS_DIR / "test_predictions.csv",
        index=False
    )

    print("Test predictions saved.")


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nBRAIN TUMOR CLASSIFICATION EVALUATION")
    print("=" * 55)

    model = load_trained_model()

    test_dataset = load_test_data()

    y_true, y_pred, probabilities = get_predictions(
        model,
        test_dataset
    )

    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=0
    )

    print("\nMODEL TEST PERFORMANCE")
    print("=" * 55)
    print(f"Test loss:     {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    metrics = calculate_metrics(
        y_true,
        y_pred,
        probabilities
    )

    metrics["test_loss"] = float(loss)
    metrics["test_accuracy"] = float(accuracy)

    with open(
        RESULTS_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

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
        probabilities
    )

    print("\nEVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 55)
    print(f"Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
