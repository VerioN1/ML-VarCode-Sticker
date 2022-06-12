

import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from classify_image import *

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers
)

class Image(BaseModel):
    image: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/image")
def read_item(base64img: Image):
    try:
        prediction = classify_image(base64img.image)
        return prediction
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="coudln't parse image")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
