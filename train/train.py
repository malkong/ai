import os
import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from models import CLIPClassifier
from dataset import get_dataloader

# colab 환경에서 실행 시 아래 주석 해제
# import sys

# sys.path.append('/content/ai')
# from models.clip import CLIPClassifier


# ==========================
# 설정
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-5
WEIGHT_DECAY = 1e-4

SAVE_DIR = "weights"
SAVE_PATH = os.path.join(SAVE_DIR, "best_model.pt")

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================
# DataLoader
# ==========================

train_loader, classes = get_dataloader(
    TRAIN_DIR,
    batch_size=BATCH_SIZE,
    shuffle=True,
    train=True
)

val_loader, _ = get_dataloader(
    VAL_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False,
    train=False
)

# ==========================
# Model
# ==========================

model = CLIPClassifier(
    num_classes=len(classes),
    freeze_backbone=False
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# ==========================
# Early Stopping
# ==========================

best_acc = 0.0
best_model = None

patience = 5
counter = 0

# ==========================
# Training
# ==========================

for epoch in range(EPOCHS):

    print("=" * 60)
    print(f"Epoch {epoch+1}/{EPOCHS}")

    ############################
    # Train
    ############################

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    loop = tqdm(train_loader)

    for batch in loop:

        images = batch["pixel_values"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        pred = outputs.argmax(1)

        train_correct += (pred == labels).sum().item()
        train_total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    train_acc = train_correct / train_total

    ############################
    # Validation
    ############################

    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch in val_loader:

            images = batch["pixel_values"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            pred = outputs.argmax(1)

            val_correct += (pred == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    scheduler.step()

    print()
    print(f"Train Loss : {train_loss/len(train_loader):.4f}")
    print(f"Train Acc  : {train_acc:.4f}")
    print(f"Val Loss   : {val_loss/len(val_loader):.4f}")
    print(f"Val Acc    : {val_acc:.4f}")

    ############################
    # Save Best
    ############################

    if val_acc > best_acc:

        best_acc = val_acc
        best_model = copy.deepcopy(model.state_dict())

        torch.save(best_model, SAVE_PATH)

        counter = 0

        print("Best Model Saved!")

    else:

        counter += 1

        print(f"EarlyStopping Counter : {counter}/{patience}")

    if counter >= patience:

        print("\nEarly Stopping")
        break

print("=" * 60)
print("Training Finished")
print(f"Best Validation Accuracy : {best_acc:.4f}")
print(f"Saved Model : {SAVE_PATH}")