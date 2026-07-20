import os

from PIL import Image
from tqdm import tqdm

from services.inference import model
from services.inference import CLASS_NAMES

from train.metrics import evaluate_metrics
from train.visualize import (
    save_confusion_matrix,
    save_misclassified,
)

TEST_DIR = "data/test"


def load_test_dataset():

    image_paths = []

    labels = []

    for label_idx, class_name in enumerate(CLASS_NAMES):

        folder = os.path.join(
            TEST_DIR,
            class_name
        )

        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):

            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                )
            ):

                image_paths.append(
                    os.path.join(
                        folder,
                        file,
                    )
                )

                labels.append(label_idx)

    return image_paths, labels


def evaluate():

    image_paths, labels = load_test_dataset()

    y_true = []

    y_pred = []

    for path, gt in tqdm(
        zip(image_paths, labels),
        total=len(image_paths),
    ):

        image = Image.open(path).convert("RGB")

        result = model.predict(
            image=image,
            class_names=CLASS_NAMES,
        )

        pred = CLASS_NAMES.index(
            result["scene"]
        )

        y_true.append(gt)

        y_pred.append(pred)

    evaluate_metrics(
        y_true,
        y_pred,
        CLASS_NAMES,
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        CLASS_NAMES,
    )

    save_misclassified(
        image_paths,
        y_true,
        y_pred,
        CLASS_NAMES,
    )


if __name__ == "__main__":
    evaluate()