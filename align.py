import os
from PIL import Image

# ==========================
# 데이터셋 최상위 폴더
# ==========================
ROOT_DIR = "data"   # <- 수정

SUPPORTED_EXT = (
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif",
    ".tiff", ".gif"
)

for current_dir, _, files in os.walk(ROOT_DIR):

    image_files = [
        f for f in files
        if f.lower().endswith(SUPPORTED_EXT)
    ]

    if not image_files:
        continue

    folder_name = os.path.basename(current_dir)

    print(f"\n===== {folder_name} =====")

    image_files.sort()

    idx = 1

    for filename in image_files:

        src_path = os.path.join(current_dir, filename)

        tmp_name = f"__tmp_{idx:03d}.jpg"
        tmp_path = os.path.join(current_dir, tmp_name)

        final_name = f"{folder_name}_{idx:03d}.jpg"
        final_path = os.path.join(current_dir, final_name)

        try:

            with Image.open(src_path) as img:

                # JPEG 저장을 위해 RGB 변환
                if img.mode != "RGB":
                    img = img.convert("RGB")

                img.save(tmp_path, "JPEG", quality=95)

            # 저장 성공했는지 확인
            if not os.path.exists(tmp_path):
                raise Exception("JPEG 저장 실패")

            # 성공한 경우에만 원본 삭제
            if os.path.abspath(src_path) != os.path.abspath(tmp_path):
                os.remove(src_path)

            os.rename(tmp_path, final_path)

            print(f"[OK] {filename} -> {final_name}")

            idx += 1

        except Exception as e:

            print(f"[ERROR] {filename}")
            print(f"        {e}")

            # tmp 파일 남았으면 삭제
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

print("\n모든 작업 완료.")