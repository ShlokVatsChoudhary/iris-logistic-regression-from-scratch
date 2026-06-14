import matplotlib.pyplot as plt
import numpy as np

#function for plotting confusion matrix
def plot_confusion_matrix(
    TN,
    FP,
    FN,
    TP,
    class_names,
    title,
    save_path=None
):
    matrix = np.array([
        [TN, FP],
        [FN, TP]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix)

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

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

#plotting the change in cost
def plot_binary_cross_entropy_cost(j_history):
    iterations = np.arange(len(j_history))

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, j_history)

    plt.xlabel("Number of iterations")
    plt.ylabel("Binary cross-entropy cost")
    plt.title("Training cost vs gradient-descent iterations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "plots/experiment_2/training_cost_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()