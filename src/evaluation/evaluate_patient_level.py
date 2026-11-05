
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data.dataset_loader import BrainTumorDatasetLoader


SEED = 42

MODEL_PATH = Path("models/classification/custom_cnn_best.keras")
RESULTS_DIR = Path("results/classification/patient_level")

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "pituitary"
]

CLASS_LABELS = [0, 1, 2]

PROBABILITY_COLUMNS = [
    "glioma_probability",
    "meningioma_probability",
    "pituitary_probability"
]


def create_directories():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    print("\nLOADING TRAINED CNN")
    print("=" * 60)

    model = tf.keras.models.load_model(MODEL_PATH)

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            "The model output does not match the number of classes."
        )

    return model


def load_test_data():
    print("\nLOADING TEST DATA")
    print("=" * 60)

    loader = BrainTumorDatasetLoader(
        processed_dir="data/processed",
        image_size=(256, 256),
        batch_size=16,
        seed=SEED
    )

    test_records = (
        loader.metadata[
            loader.metadata["split"] == "test"
        ]
        .copy()
        .reset_index(drop=True)
    )

    if test_records.empty:
        raise ValueError("The test partition is empty.")

    if test_records["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs found in test metadata.")

    test_dataset = loader.get_classification_datasets()["test"]

    print(f"Test MRI images: {len(test_records)}")
    print(f"Unique test patients: {test_records['patient_id'].nunique()}")

    return test_dataset, test_records, loader.class_to_index


def generate_predictions(model, test_dataset, test_records, class_to_index):
    print("\nGENERATING IMAGE-LEVEL PREDICTIONS")
    print("=" * 60)

    probabilities = model.predict(
        test_dataset,
        verbose=1
    )

    if len(probabilities) != len(test_records):
        raise ValueError(
            "Prediction count does not match test metadata count."
        )

    if probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            "Unexpected number of model output classes."
        )

    true_labels = np.array(
        [
            class_to_index[label]
            for label in test_records["label"]
        ],
        dtype=np.int32
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1
    )

    image_predictions = test_records[
        ["sample_id", "patient_id", "label"]
    ].copy()

    image_predictions["true_label"] = [
        CLASS_NAMES[index] for index in true_labels
    ]

    image_predictions["predicted_label"] = [
        CLASS_NAMES[index] for index in predicted_labels
    ]

    for index, column in enumerate(PROBABILITY_COLUMNS):
        image_predictions[column] = probabilities[:, index]

    image_predictions["correct"] = (
        image_predictions["true_label"]
        == image_predictions["predicted_label"]
    )

    return image_predictions, probabilities


def aggregate_patient_predictions(image_predictions):
    print("\nAGGREGATING PREDICTIONS BY PATIENT")
    print("=" * 60)

    label_counts = (
        image_predictions.groupby("patient_id")["true_label"]
        .nunique()
    )

    if (label_counts > 1).any():
        raise ValueError(
            "Some patients have multiple ground-truth classes. "
            "Patient-level aggregation requires one consistent "
            "class per patient."
        )

    patient_predictions = (
        image_predictions.groupby("patient_id", as_index=False)
        .agg(
            true_label=("true_label", "first"),
            number_of_images=("sample_id", "count"),
            correct_image_predictions=("correct", "sum"),
            **{
                column: (column, "mean")
                for column in PROBABILITY_COLUMNS
            }
        )
    )

    patient_predictions["predicted_label"] = [
        CLASS_NAMES[index]
        for index in np.argmax(
            patient_predictions[PROBABILITY_COLUMNS].to_numpy(),
            axis=1
        )
    ]

    patient_predictions["correct"] = (
        patient_predictions["true_label"]
        == patient_predictions["predicted_label"]
    )

    print(f"Unique patients: {len(patient_predictions)}")
    print(
        f"Correct patient predictions: "
        f"{patient_predictions['correct'].sum()}"
    )
    print(
        f"Incorrect patient predictions: "
        f"{(~patient_predictions['correct']).sum()}"
    )

    return patient_predictions


def calculate_specificity(y_true, y_pred):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_LABELS
    )

    specificities = {}

    for index, class_name in enumerate(CLASS_NAMES):
        true_negative = (
            matrix.sum()
            - matrix[index, :].sum()
            - matrix[:, index].sum()
            + matrix[index, index]
        )

        false_positive = (
            matrix[:, index].sum()
            - matrix[index, index]
        )

        denominator = true_negative + false_positive

        specificities[class_name] = (
            true_negative / denominator
            if denominator > 0
            else 0.0
        )

    return matrix, specificities


def evaluate_patient_predictions(patient_predictions):
    y_true = np.array(
        [
            CLASS_NAMES.index(label)
            for label in patient_predictions["true_label"]
        ]
    )

    y_pred = np.array(
        [
            CLASS_NAMES.index(label)
            for label in patient_predictions["predicted_label"]
        ]
    )

    probabilities = patient_predictions[
        PROBABILITY_COLUMNS
    ].to_numpy()

    metrics = {
        "number_of_patients": len(patient_predictions),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
    }

    matrix, specificities = calculate_specificity(
        y_true,
        y_pred
    )

    metrics["specificity_macro"] = np.mean(
        list(specificities.values())
    )

    if len(np.unique(y_true)) == len(CLASS_NAMES):
        metrics["roc_auc_macro_ovr"] = roc_auc_score(
            y_true,
            probabilities,
            labels=CLASS_LABELS,
            multi_class="ovr",
            average="macro"
        )
    else:
        metrics["roc_auc_macro_ovr"] = np.nan

    return y_true, y_pred, metrics, matrix, specificities


def save_results(
    patient_predictions,
    y_true,
    y_pred,
    metrics,
    specificities
):
    predictions_path = (
        RESULTS_DIR / "patient_predictions.csv"
    )

    patient_predictions.to_csv(
        predictions_path,
        index=False
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_path = (
        RESULTS_DIR / "patient_classification_report.csv"
    )

    pd.DataFrame(report).transpose().to_csv(report_path)

    metrics_path = (
        RESULTS_DIR / "patient_evaluation_metrics.csv"
    )

    pd.DataFrame([metrics]).to_csv(
        metrics_path,
        index=False
    )

    specificity_path = (
        RESULTS_DIR / "patient_specificity.csv"
    )

    pd.DataFrame(
        list(specificities.items()),
        columns=["class", "specificity"]
    ).to_csv(
        specificity_path,
        index=False
    )

    return (
        predictions_path,
        report_path,
        metrics_path,
        specificity_path
    )


def plot_patient_confusion_matrix(matrix):
    figure, axis = plt.subplots(figsize=(8, 6))

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues"
    )

    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted Patient Class",
        ylabel="True Patient Class",
        title="Patient-Level Confusion Matrix"
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=30,
        ha="right"
    )

    threshold = matrix.max() / 2.0

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                )
            )

    figure.tight_layout()

    path = RESULTS_DIR / "patient_confusion_matrix.png"

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    return path


def main():
    tf.keras.utils.set_random_seed(SEED)

    create_directories()

    model = load_model()

    test_dataset, test_records, class_to_index = load_test_data()

    image_predictions, _ = generate_predictions(
        model,
        test_dataset,
        test_records,
        class_to_index
    )

    image_predictions.to_csv(
        RESULTS_DIR / "image_predictions_with_patient_ids.csv",
        index=False
    )

    patient_predictions = aggregate_patient_predictions(
        image_predictions
    )

    (
        y_true,
        y_pred,
        metrics,
        matrix,
        specificities
    ) = evaluate_patient_predictions(patient_predictions)

    (
        predictions_path,
        report_path,
        metrics_path,
        specificity_path
    ) = save_results(
        patient_predictions,
        y_true,
        y_pred,
        metrics,
        specificities
    )

    confusion_path = plot_patient_confusion_matrix(matrix)

    print("\nPATIENT-LEVEL CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("\nPATIENT-LEVEL METRICS")
    print("=" * 60)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nPATIENT-LEVEL SPECIFICITY")
    print("=" * 60)

    for class_name, value in specificities.items():
        print(f"{class_name}: {value:.4f}")

    print("\nSAVED FILES")
    print("=" * 60)
    print(f"Patient predictions: {predictions_path}")
    print(f"Classification report: {report_path}")
    print(f"Evaluation metrics: {metrics_path}")
    print(f"Specificity: {specificity_path}")
    print(f"Confusion matrix: {confusion_path}")
    print(
        "Image predictions with patient IDs: "
        f"{RESULTS_DIR / 'image_predictions_with_patient_ids.csv'}"
    )

    print("\nPATIENT-LEVEL EVALUATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
