
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.data.dataset_loader import BrainTumorDatasetLoader
from src.models.unet import build_unet


SEED = 42
EPOCHS = 50
BATCH_SIZE = 4

MODEL_DIR = Path("models/segmentation")
RESULTS_DIR = Path("results/segmentation/flip_only_experiment")

MODEL_PATH = MODEL_DIR / "unet_flip_only_best.keras"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"
PLOT_PATH = RESULTS_DIR / "training_curves.png"


def create_directories():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def augment_sample(image, mask):
    should_flip = tf.random.uniform(()) > 0.5

    image, mask = tf.cond(
        should_flip,
        lambda: (
            tf.image.flip_left_right(image),
            tf.image.flip_left_right(mask)
        ),
        lambda: (image, mask)
    )

    return image, mask


def prepare_augmented_training_dataset(dataset):
    dataset = dataset.unbatch()

    dataset = dataset.map(
        augment_sample,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def plot_training_history(history):
    history_df = pd.DataFrame(history.history)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        history_df["dice_coefficient"],
        label="Training Dice"
    )

    axes[0].plot(
        history_df["val_dice_coefficient"],
        label="Validation Dice"
    )

    axes[0].set_title("Flip-Only U-Net Dice")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Dice")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        history_df["loss"],
        label="Training Loss"
    )

    axes[1].plot(
        history_df["val_loss"],
        label="Validation Loss"
    )

    axes[1].set_title("Flip-Only U-Net Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    figure.suptitle(
        "Horizontal Flip-Only U-Net Training",
        fontsize=14
    )

    figure.tight_layout()

    figure.savefig(
        PLOT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(figure)

    history_df.index = history_df.index + 1
    history_df.index.name = "epoch"
    history_df.to_csv(HISTORY_PATH)


def main():
    tf.keras.utils.set_random_seed(SEED)

    create_directories()

    print("\nLOADING SEGMENTATION DATASET")
    print("=" * 60)

    loader = BrainTumorDatasetLoader(
        processed_dir="data/processed",
        image_size=(256, 256),
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    datasets = loader.get_segmentation_datasets()

    train_dataset = prepare_augmented_training_dataset(
        datasets["train"]
    )

    validation_dataset = datasets["validation"]

    print("\nAUGMENTATION CONFIGURATION")
    print("=" * 60)
    print("Horizontal flipping: 50% probability")
    print("Rotation: disabled")
    print("Validation augmentation: disabled")

    print("\nBUILDING U-NET")
    print("=" * 60)

    model = build_unet(input_shape=(256, 256, 1))

    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.CSVLogger(
            str(RESULTS_DIR / "training_log.csv")
        )
    ]

    print("\nSTARTING FLIP-ONLY TRAINING")
    print("=" * 60)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    best_epoch = (
        history.history["val_loss"].index(
            min(history.history["val_loss"])
        ) + 1
    )

    best_val_loss = min(history.history["val_loss"])
    best_val_dice = max(history.history["val_dice_coefficient"])

    print("\nFLIP-ONLY TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation Dice: {best_val_dice:.6f}")

    plot_training_history(history)

    print("\nSAVED FILES")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"History: {HISTORY_PATH}")
    print(f"Training log: {RESULTS_DIR / 'training_log.csv'}")
    print(f"Training curves: {PLOT_PATH}")


if __name__ == "__main__":
    main()
