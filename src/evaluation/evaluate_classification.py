
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
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from src.data.dataset_loader import BrainTumorDatasetLoader


SEED = 42

MODEL_PATH = Path("models/classification/custom_cnn_best.keras")
RESULTS_DIR = Path("results/classification")

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "pituitary"
]

CLASS_LABELS = [0, 1, 2]


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

    model.summary()

    return model


def load_test_data():
    print("\nLOADING TEST DATASET")
    print("=" * 60)

    loader = BrainTumorDatasetLoader(
        processed_dir="data/processed",
        image_size=(256, 256),
        batch_size=16,
        seed=SEED
    )

    datasets = loader.get_classification_datasets()

    test_dataset = datasets["test"]

    print(f"Test images: {len(loader.metadata[loader.metadata['split'] == 'test'])}")
    print(f"Classes: {CLASS_NAMES}")

    return test_dataset


def get_predictions(model, test_dataset):
    print("\nGENERATING PREDICTIONS")
    print("=" * 60)

    true_labels = []

    for _, labels in test_dataset:
        true_labels.extend(labels.numpy())

    true_labels = np.array(true_labels, dtype=np.int32)

    probabilities = model.predict(
        test_dataset,
        verbose=1
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1
    )

    if probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output classes, "
            f"but model returned {probabilities.shape[1]}."
        )

    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            "Number of true labels and predictions does not match."
        )

    return true_labels, predicted_labels, probabilities


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

        false_positive = matrix[:, index].sum() - matrix[index, index]

        denominator = true_negative + false_positive

        specificity = (
            true_negative / denominator
            if denominator > 0
            else 0.0
        )

        specificities[class_name] = specificity

    return matrix, specificities


def calculate_metrics(y_true, y_pred, probabilities):
    accuracy = accuracy_score(y_true, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    precision_macro = precision_score(
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

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    matrix, specificities = calculate_specificity(
        y_true,
        y_pred
    )

    specificity_macro = np.mean(
        list(specificities.values())
    )

    y_true_binary = label_binarize(
        y_true,
        classes=CLASS_LABELS
    )

    roc_auc_macro = roc_auc_score(
        y_true_binary,
        probabilities,
        average="macro",
        multi_class="ovr"
    )

    roc_auc_weighted = roc_auc_score(
        y_true_binary,
        probabilities,
        average="weighted",
        multi_class="ovr"
    )

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "specificity_macro": specificity_macro,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted
    }

    return metrics, matrix, specificities


def save_classification_report(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = RESULTS_DIR / "classification_report.csv"

    report_df.to_csv(report_path)

    print("\nCLASSIFICATION REPORT")
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

    return report_path


def save_metrics(metrics, specificities):
    metrics_path = RESULTS_DIR / "evaluation_metrics.csv"

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    specificity_path = RESULTS_DIR / "class_specificity.csv"

    specificity_df = pd.DataFrame(
        list(specificities.items()),
        columns=["class", "specificity"]
    )

    specificity_df.to_csv(
        specificity_path,
        index=False
    )

    print("\nOVERALL EVALUATION METRICS")
    print("=" * 60)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nCLASS SPECIFICITY")
    print("=" * 60)

    for class_name, value in specificities.items():
        print(f"{class_name}: {value:.4f}")

    return metrics_path, specificity_path


def plot_confusion_matrix(matrix):
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
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Custom CNN Confusion Matrix"
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
                format(matrix[row, column], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black"
            )

    figure.tight_layout()

    path = RESULTS_DIR / "confusion_matrix.png"

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(figure)

    return path


def plot_roc_curves(y_true, probabilities):
    y_true_binary = label_binarize(
        y_true,
        classes=CLASS_LABELS
    )

    figure, axis = plt.subplots(figsize=(9, 7))

    for index, class_name in enumerate(CLASS_NAMES):
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

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    axis.set_title("Multiclass ROC Curves")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    axis.grid(True)

    figure.tight_layout()

    path = RESULTS_DIR / "roc_curves.png"

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(figure)

    return path


def save_predictions(y_true, y_pred, probabilities):
    predictions_df = pd.DataFrame({
        "true_label": [
            CLASS_NAMES[index] for index in y_true
        ],
        "predicted_label": [
            CLASS_NAMES[index] for index in y_pred
        ],
        "glioma_probability": probabilities[:, 0],
        "meningioma_probability": probabilities[:, 1],
        "pituitary_probability": probabilities[:, 2]
    })

    path = RESULTS_DIR / "test_predictions.csv"

    predictions_df.to_csv(
        path,
        index=False
    )

    return path


def main():
    tf.keras.utils.set_random_seed(SEED)

    create_directories()

    model = load_model()

    test_dataset = load_test_data()

    y_true, y_pred, probabilities = get_predictions(
        model,
        test_dataset
    )

    metrics, matrix, specificities = calculate_metrics(
        y_true,
        y_pred,
        probabilities
    )

    report_path = save_classification_report(
        y_true,
        y_pred
    )

    metrics_path, specificity_path = save_metrics(
        metrics,
        specificities
    )

    confusion_path = plot_confusion_matrix(matrix)

    roc_path = plot_roc_curves(
        y_true,
        probabilities
    )

    predictions_path = save_predictions(
        y_true,
        y_pred,
        probabilities
    )

    print("\nSAVED EVALUATION FILES")
    print("=" * 60)
    print(f"Classification report: {report_path}")
    print(f"Evaluation metrics: {metrics_path}")
    print(f"Class specificity: {specificity_path}")
    print(f"Confusion matrix: {confusion_path}")
    print(f"ROC curves: {roc_path}")
    print(f"Test predictions: {predictions_path}")

    print("\nCLASSIFICATION EVALUATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
