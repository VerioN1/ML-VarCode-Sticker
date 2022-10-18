import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
from classify_image import *
import tensorflow as tf

from train_model_ml import train_model_func

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)


class Image(BaseModel):
    image: str


model = tf.keras.models.load_model("my_model")
# optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model.compile(optimizer=optimizer,
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/image")
def read_item(base64img: Image):
    try:
        prediction = classify_image(base64img.image, model)
        return prediction
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="coudln't parse image")


@app.post("/train-model")
def train_model(base64img: Image, isFrozen: str = Query(None, description="Yes or No are the only valid inputs to isFrozen")):
    try:
        is_frozen = 0 if isFrozen.lower() == "yes" else 1
        train_model_status = train_model_func(model, base64img.image, is_frozen)
        model.save('my_model')
        return "model is trained" if train_model_status else "model rejected this photo"
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
