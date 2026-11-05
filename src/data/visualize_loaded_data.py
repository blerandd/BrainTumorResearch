
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset_loader import BrainTumorDatasetLoader


def visualize_classification(loader):
    datasets = loader.get_classification_datasets()
    images, labels = next(iter(datasets["train"]))

    images = images.numpy()
    labels = labels.numpy()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for index, ax in enumerate(axes.flat):
        ax.imshow(images[index].squeeze(), cmap="gray")
        ax.set_title(loader.CLASSES[labels[index]])
        ax.axis("off")

    fig.suptitle("MRI Images Loaded by TensorFlow", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        "results/mri_loaded_batch.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


def visualize_segmentation(loader):
    datasets = loader.get_segmentation_datasets()
    images, masks = next(iter(datasets["train"]))

    images = images.numpy()
    masks = masks.numpy()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for index, ax in enumerate(axes.flat):
        image = images[index].squeeze()
        mask = masks[index].squeeze()

        ax.imshow(image, cmap="gray")
        ax.imshow(
            np.ma.masked_where(mask == 0, mask),
            cmap="autumn",
            alpha=0.55,
            vmin=0,
            vmax=1
        )
        ax.set_title(f"Sample {index + 1}")
        ax.axis("off")

    fig.suptitle("MRI Images with Ground-Truth Tumor Masks", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        "results/mri_mask_batch.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    loader = BrainTumorDatasetLoader()

    visualize_classification(loader)
    visualize_segmentation(loader)
