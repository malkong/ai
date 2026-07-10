from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from PIL import Image

from models.clip import CLIPInference

app = FastAPI()

model = CLIPInference()

CLASS_NAMES = [
    "Hospital Reception": "병원 접수대",
    "Pharmacy Counter": "약국 계산대",
    "Cafe Counter": "카페 주문대",
    "Restaurant Counter": "식당 주문대",
    "Convenience Store Counter": "편의점 계산대",
    "Bus Entrance": "버스 승차문",
    "Subway Gate": "지하철 개찰구"
] 

@app.get("/")
def root():

    return {
        "message": "AAC CLIP Server"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image = Image.open(file.file).convert("RGB")

    result = model.predict(
        image=image,
        class_names=CLASS_NAMES
    )

    return result