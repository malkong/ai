import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from models import CLIPClassifier
from dataset import get_dataloader

# colab 환경에서 실행 시 아래 주석 해제
# import sys

# sys.path.append('/content/ai')
# from models.clip import CLIPClassifier


# ===============================
# 설정
# ===============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DIR = "dataset/test"

WEIGHT = "weights/best_model.pt"

RESULT_DIR = "results"
MIS_DIR = os.path.join(RESULT_DIR, "misclassified")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MIS_DIR, exist_ok=True)

# ===============================
# Data
# ===============================

test_loader, classes = get_dataloader(
    TEST_DIR,
    batch_size=16,
    shuffle=False,
    train=False
)

# ===============================
# Model
# ===============================

model = CLIPClassifier(
    num_classes=len(classes),
    freeze_backbone=False
)

model.load_state_dict(
    torch.load(
        WEIGHT,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

# ===============================
# Evaluation
# ===============================

y_true = []
y_pred = []

with torch.no_grad():

    for batch in test_loader:

        images = batch["pixel_values"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(images)

        pred = outputs.argmax(1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

        paths = batch["path"]

        for i in range(len(paths)):

            if pred[i].item() != labels[i].item():

                filename = os.path.basename(paths[i])

                shutil.copy(
                    paths[i],
                    os.path.join(
                        MIS_DIR,
                        f"{classes[labels[i]]}_pred_{classes[pred[i]]}_{filename}"
                    )
                )

# ===============================
# Metrics
# ===============================

acc = accuracy_score(y_true, y_pred)

print("=" * 60)
print(f"Accuracy : {acc:.4f}")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4
    )
)

# ===============================
# Confusion Matrix
# ===============================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 7))

plt.imshow(cm)

plt.xticks(
    np.arange(len(classes)),
    classes,
    rotation=45
)

plt.yticks(
    np.arange(len(classes)),
    classes
)

plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

print()
print("Confusion Matrix 저장 완료")
print(f"Misclassified : {len(os.listdir(MIS_DIR))}")