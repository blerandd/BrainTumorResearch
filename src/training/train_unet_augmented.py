
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.data.dataset_loader import BrainTumorDatasetLoader
from src.models.unet import build_unet


SEED = 42
EPOCHS = 50
BATCH_SIZE = 4
ROTATION_RANGE = 10.0

MODEL_DIR = Path("models/segmentation")
RESULTS_DIR = Path("results/segmentation/augmented_experiment")

MODEL_PATH = MODEL_DIR / "unet_augmented_best.keras"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"
PLOT_PATH = RESULTS_DIR / "training_curves.png"


def create_directories():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def apply_random_rotation(image, mask):
    angle = tf.random.uniform(
        shape=(),
        minval=-ROTATION_RANGE,
        maxval=ROTATION_RANGE
    )

    angle = angle * (3.141592653589793 / 180.0)

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    center_x = tf.cast(width - 1, tf.float32) / 2.0
    center_y = tf.cast(height - 1, tf.float32) / 2.0

    cosine = tf.cos(angle)
    sine = tf.sin(angle)

    transform = tf.stack([
        cosine,
        -sine,
        center_x - cosine * center_x + sine * center_y,
        sine,
        cosine,
        center_y - sine * center_x - cosine * center_y,
        0.0,
        0.0
    ])

    transform = tf.reshape(transform, (1, 8))

    output_shape = tf.stack([height, width])

    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, axis=0),
        transforms=transform,
        output_shape=output_shape,
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0
    )[0]

    mask = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(mask, axis=0),
        transforms=transform,
        output_shape=output_shape,
        interpolation="NEAREST",
        fill_mode="CONSTANT",
        fill_value=0.0
    )[0]

    image.set_shape((256, 256, 1))
    mask.set_shape((256, 256, 1))

    mask = tf.cast(mask > 0.5, tf.float32)

    return image, mask


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

    image, mask = apply_random_rotation(image, mask)

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

    axes[0].set_title("Augmented U-Net Dice Coefficient")
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

    axes[1].set_title("Augmented U-Net Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    figure.suptitle(
        "Augmented U-Net Training and Validation Performance",
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

    print("\nLOADING BRAIN TUMOR SEGMENTATION DATASET")
    print("=" * 60)

    loader = BrainTumorDatasetLoader(
        processed_dir="data/processed",
        image_size=(256, 256),
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    datasets = loader.get_segmentation_datasets()

    print("\nPREPARING AUGMENTED TRAINING DATA")
    print("=" * 60)

    train_dataset = prepare_augmented_training_dataset(
        datasets["train"]
    )

    validation_dataset = datasets["validation"]

    print("Training augmentation:")
    print("- Random horizontal flipping")
    print("- Random rotations between -10 and +10 degrees")
    print("- Bilinear interpolation for MRI images")
    print("- Nearest-neighbor interpolation for masks")
    print("- Validation data remains unchanged")

    print("\nBUILDING U-NET MODEL")
    print("=" * 60)

    model = build_unet(
        input_shape=(256, 256, 1)
    )

    model.summary()

    print("\nMODEL CONFIGURATION")
    print("=" * 60)
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

    print("\nSTARTING AUGMENTED U-NET TRAINING")
    print("=" * 60)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("\nAUGMENTED U-NET TRAINING COMPLETED")
    print("=" * 60)

    best_epoch = (
        history.history["val_loss"].index(
            min(history.history["val_loss"])
        ) + 1
    )

    best_val_loss = min(
        history.history["val_loss"]
    )

    best_val_dice = max(
        history.history["val_dice_coefficient"]
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation Dice: {best_val_dice:.4f}")

    plot_training_history(history)

    print("\nSAVED FILES")
    print("=" * 60)
    print(f"Best augmented model: {MODEL_PATH}")
    print(f"Training history: {HISTORY_PATH}")
    print(f"Training log: {RESULTS_DIR / 'training_log.csv'}")
    print(f"Training curves: {PLOT_PATH}")

    print("\nAUGMENTED U-NET EXPERIMENT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()