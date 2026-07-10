from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File

app = FastAPI()


@app.get("/")
def root():

    return {
        "message": "AAC CLIP Server"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # TODO
    # Save Image
    # Run CLIP

    return {
        "scene": "Hospital Reception"
    }