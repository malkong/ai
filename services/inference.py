# services/inference.py

from models.clip import CLIPModel

# 서버가 시작될 때 한 번만 모델 로드
model = CLIPModel()

CLASS_NAMES = [
    "Hospital Reception",
    "Pharmacy Counter",
    "Cafe Counter",
    "Restaurant Counter",
    "Convenience Store Counter",
    "Bus Entrance",
    "Subway Gate"
]


def predict_image(image):
    """
    PIL.Image를 입력받아 CLIP 추론 수행
    """

    result = model.predict(
        image=image,
        class_names=CLASS_NAMES
    )

    return result
