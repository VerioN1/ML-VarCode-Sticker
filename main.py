from typing import Optional

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from classify_image import *

app = FastAPI()


class Image(BaseModel):
    image: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/image")
def read_item(base64img: Image):
    prediction = classify_image(base64img.image)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
