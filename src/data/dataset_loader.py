
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


class BrainTumorDatasetLoader:

    CLASSES = ["glioma", "meningioma", "pituitary"]

    def __init__(
        self,
        processed_dir="data/processed",
        image_size=(256, 256),
        batch_size=16,
        seed=42
    ):
        self.processed_dir = Path(processed_dir)
        self.metadata_path = self.processed_dir / "splits.csv"
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed
        self.class_to_index = {
            name: index for index, name in enumerate(self.CLASSES)
        }

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        self.metadata = pd.read_csv(self.metadata_path)
        self._validate_metadata()

    def _validate_metadata(self):
        required_columns = {
            "sample_id",
            "patient_id",
            "label",
            "image_path",
            "mask_path",
            "split"
        }

        missing_columns = required_columns - set(self.metadata.columns)

        if missing_columns:
            raise ValueError(
                f"Missing metadata columns: {sorted(missing_columns)}"
            )

        if self.metadata.isnull().any().any():
            raise ValueError("Metadata contains missing values.")

        unknown_classes = set(self.metadata["label"]) - set(self.CLASSES)

        if unknown_classes:
            raise ValueError(
                f"Unknown classes found: {sorted(unknown_classes)}"
            )

        expected_splits = {"train", "validation", "test"}
        actual_splits = set(self.metadata["split"])

        if actual_splits != expected_splits:
            raise ValueError(
                f"Expected splits {expected_splits}, got {actual_splits}"
            )

        for column in ["image_path", "mask_path"]:
            for relative_path in self.metadata[column]:
                full_path = self.processed_dir / relative_path

                if not full_path.is_file():
                    raise FileNotFoundError(
                        f"Dataset file not found: {full_path}"
                    )

        patient_split_counts = (
            self.metadata.groupby("patient_id")["split"].nunique()
        )

        if (patient_split_counts > 1).any():
            raise ValueError(
                "Patient leakage detected across dataset partitions."
            )

    def _load_image(self, image_path):
        image_bytes = tf.io.read_file(image_path)

        image = tf.io.decode_png(
            image_bytes,
            channels=1,
            dtype=tf.uint16
        )

        image = tf.cast(image, tf.float32) / 65535.0

        image = tf.image.resize(
            image,
            self.image_size,
            method="bilinear"
        )

        image.set_shape(
            (self.image_size[0], self.image_size[1], 1)
        )

        return image

    def _load_mask(self, mask_path):
        mask_bytes = tf.io.read_file(mask_path)

        mask = tf.io.decode_png(
            mask_bytes,
            channels=1,
            dtype=tf.uint8
        )

        mask = tf.cast(mask > 0, tf.float32)

        mask = tf.image.resize(
            mask,
            self.image_size,
            method="nearest"
        )

        mask = tf.cast(mask > 0.5, tf.float32)

        mask.set_shape(
            (self.image_size[0], self.image_size[1], 1)
        )

        return mask

    def _build_classification_dataset(self, split):
        records = self.metadata[
            self.metadata["split"] == split
        ].copy()

        image_paths = [
            str(self.processed_dir / path)
            for path in records["image_path"]
        ]

        labels = np.array(
            [
                self.class_to_index[label]
                for label in records["label"]
            ],
            dtype=np.int32
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            (image_paths, labels)
        )

        def load_sample(image_path, label):
            image = self._load_image(image_path)
            return image, label

        dataset = dataset.map(
            load_sample,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if split == "train":
            dataset = dataset.shuffle(
                buffer_size=len(records),
                seed=self.seed,
                reshuffle_each_iteration=True
            )

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _build_segmentation_dataset(self, split):
        records = self.metadata[
            self.metadata["split"] == split
        ].copy()

        image_paths = [
            str(self.processed_dir / path)
            for path in records["image_path"]
        ]

        mask_paths = [
            str(self.processed_dir / path)
            for path in records["mask_path"]
        ]

        dataset = tf.data.Dataset.from_tensor_slices(
            (image_paths, mask_paths)
        )

        def load_sample(image_path, mask_path):
            image = self._load_image(image_path)
            mask = self._load_mask(mask_path)

            return image, mask

        dataset = dataset.map(
            load_sample,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if split == "train":
            dataset = dataset.shuffle(
                buffer_size=len(records),
                seed=self.seed,
                reshuffle_each_iteration=True
            )

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_classification_datasets(self):
        return {
            split: self._build_classification_dataset(split)
            for split in ["train", "validation", "test"]
        }

    def get_segmentation_datasets(self):
        return {
            split: self._build_segmentation_dataset(split)
            for split in ["train", "validation", "test"]
        }

    def print_summary(self):
        print("\nBRAIN TUMOR DATASET LOADER")
        print("=" * 55)

        print(f"Total MRI images: {len(self.metadata)}")
        print(
            f"Unique patients: "
            f"{self.metadata['patient_id'].nunique()}"
        )

        print(f"Image size: {self.image_size}")
        print(f"Batch size: {self.batch_size}")

        print("\nCLASS MAPPING")
        for class_name, index in self.class_to_index.items():
            print(f"{index}: {class_name}")

        print("\nPARTITION SUMMARY")

        summary = self.metadata.groupby("split").agg(
            images=("sample_id", "count"),
            patients=("patient_id", "nunique")
        )

        print(summary)

        print("\nCLASSIFICATION DATASET CHECK")

        classification = self.get_classification_datasets()

        for split, dataset in classification.items():
            images, labels = next(iter(dataset))

            print(
                f"{split}: "
                f"images={images.shape}, "
                f"labels={labels.shape}"
            )

        print("\nSEGMENTATION DATASET CHECK")

        segmentation = self.get_segmentation_datasets()

        for split, dataset in segmentation.items():
            images, masks = next(iter(dataset))

            print(
                f"{split}: "
                f"images={images.shape}, "
                f"masks={masks.shape}"
            )

        print("\nDATASET LOADING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    loader = BrainTumorDatasetLoader()
    loader.print_summary()
