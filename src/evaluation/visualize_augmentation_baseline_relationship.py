from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


INPUT = Path(
    "results/segmentation/statistical_analysis/"
    "augmentation_morphology/patient_augmentation_rankings.csv"
)

OUTPUT_DIR = Path(
    "results/segmentation/statistical_analysis/"
    "augmentation_morphology/visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

rho, p_value = spearmanr(
    df["baseline_dice"],
    df["dice_improvement"]
)

fig, ax = plt.subplots(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="baseline_dice",
    y="dice_improvement",
    hue="label",
    s=100,
    alpha=0.85,
    ax=ax
)

sns.regplot(
    data=df,
    x="baseline_dice",
    y="dice_improvement",
    scatter=False,
    color="black",
    line_kws={"linestyle": "--"},
    ax=ax
)

ax.axhline(0, linestyle=":", linewidth=1.5)

ax.set_title(
    "Baseline Segmentation Performance vs Augmentation Improvement\n"
    f"Spearman rho = {rho:.3f}, p = {p_value:.4f}"
)

ax.set_xlabel("Baseline Patient-Level Dice")
ax.set_ylabel("Dice Improvement (Augmented - Baseline)")

ax.legend(title="Tumor Class")
ax.grid(alpha=0.25)

plt.tight_layout()

output_path = OUTPUT_DIR / "baseline_dice_vs_augmentation_improvement.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("Visualization saved:", output_path)
print("Spearman rho:", round(rho, 4))
print("P-value:", round(p_value, 6))