import os
import shutil
from sklearn.model_selection import train_test_split

# ======================
# 설정
# ======================
DATA_DIR = "data"       # 원본 데이터셋
OUTPUT_DIR = "dataset"  # 분할 결과 저장 폴더

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif", ".avif")

# ======================
# 폴더 생성
# ======================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# ======================
# 클래스별 분할
# ======================
classes = sorted(
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
)

print("=" * 50)

for cls in classes:

    class_dir = os.path.join(DATA_DIR, cls)

    images = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]

    images.sort()

    train_files, temp_files = train_test_split(
        images,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_files, test_files = train_test_split(
        temp_files,
        train_size=val_ratio_adjusted,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    split_dict = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split, file_list in split_dict.items():

        dst_dir = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(dst_dir, exist_ok=True)

        for file in file_list:
            shutil.copy2(
                os.path.join(class_dir, file),
                os.path.join(dst_dir, file)
            )

    print(f"{cls:20s} | Total {len(images):3d} | Train {len(train_files):3d} | Val {len(val_files):3d} | Test {len(test_files):3d}")

print("=" * 50)
print("Dataset split 완료!")