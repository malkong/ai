from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File

from PIL import Image

from services.inference import predict_image

app = FastAPI()


@app.get("/")
def root():
    return {
        "message": "AAC CLIP API"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    result = predict_image(image)

    return result