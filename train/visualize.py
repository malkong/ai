import os

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_dir="results",
):

    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(8,8))

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "confusion_matrix.png"),
        dpi=300
    )

    plt.close()


def save_misclassified(
    image_paths,
    y_true,
    y_pred,
    class_names,
    save_path="results/misclassified.csv",
):

    import pandas as pd

    rows = []

    for path, gt, pred in zip(
        image_paths,
        y_true,
        y_pred,
    ):

        if gt != pred:

            rows.append(
                {
                    "image": path,
                    "ground_truth": class_names[gt],
                    "prediction": class_names[pred],
                }
            )

    df = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)

    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )