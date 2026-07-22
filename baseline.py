import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------
# 설정
# ---------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DIR = "dataset/test"

SAVE_DIR = "baseline_results"
MIS_DIR = os.path.join(SAVE_DIR, "misclassified")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MIS_DIR, exist_ok=True)

# ---------------------------------------------------
# 클래스
# ---------------------------------------------------

classes = [
    "Cafe",
    "Convenience Store",
    "Hospital",
    "Pharmacy",
    "Public Transport",
    "Restaurant"
]

prompts = [
    "a photo taken inside a cafe",
    "a photo taken inside a convenience store",
    "a photo taken inside a hospital",
    "a photo taken inside a pharmacy",
    "a photo taken inside public transportation",
    "a photo taken inside a restaurant"
]

# ---------------------------------------------------
# 모델
# ---------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(DEVICE)

model.eval()

# ---------------------------------------------------
# Text Embedding
# ---------------------------------------------------

text_inputs = processor(
    text=prompts,
    return_tensors="pt",
    padding=True
)

text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}

with torch.no_grad():

    text_outputs = model.get_text_features(**text_inputs)

    text_features = text_outputs.pooler_output
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

y_true = []
y_pred = []

for gt, cls in enumerate(classes):

    folder = os.path.join(TEST_DIR, cls)

    for img_name in os.listdir(folder):

        path = os.path.join(folder, img_name)

        try:
            image = Image.open(path).convert("RGB")
        except:
            continue

        image_inputs = processor(
            images=image,
            return_tensors="pt"
        )

        image_inputs = {
            k: v.to(DEVICE)
            for k, v in image_inputs.items()
        }

        with torch.no_grad():

            image_outputs = model.get_image_features(**image_inputs)

            image_feature = image_outputs.pooler_output
            image_features = image_feature / image_feature.norm(dim=-1, keepdim=True)

            similarity = image_features @ text_features.T

            pred = similarity.argmax(dim=-1).item()

        y_true.append(gt)
        y_pred.append(pred)

        if pred != gt:

            shutil.copy(
                path,
                os.path.join(
                    MIS_DIR,
                    f"{cls}_pred_{classes[pred]}_{img_name}"
                )
            )

# ---------------------------------------------------
# 결과 출력
# ---------------------------------------------------

print("=" * 50)

acc = accuracy_score(y_true, y_pred)

print(f"Accuracy : {acc:.4f}")

print("=" * 50)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4
    )
)

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,7))

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
        SAVE_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

print()

print("Confusion Matrix 저장 완료")

print("Misclassified :", len(os.listdir(MIS_DIR)))