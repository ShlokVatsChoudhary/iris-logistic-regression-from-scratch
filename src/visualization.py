import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#function for plotting confusion matrix
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    TN,
    FP,
    FN,
    TP,
    class_names,
    title,
    save_folder=None,
    image_name=None
):
    matrix = np.array([
        [TN, FP],
        [FN, TP]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)

    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax, label="Count")
    plt.tight_layout()

    if save_folder is not None and image_name is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        save_path = save_folder / f"{image_name}.png"

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

#plotting the change in cost
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_binary_cross_entropy_cost(
    j_history,
    save_folder,
    image_name
):
    iterations = np.arange(len(j_history))

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, j_history)

    plt.xlabel("Number of iterations")
    plt.ylabel("Binary cross-entropy cost")
    plt.title("Training cost vs gradient-descent iterations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    save_path = save_folder / f"{image_name}.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()