import base64

import cv2  # OpenCV Library
import numpy as np

from check_contour import is_image_contoured
from read_file import get_labels, get_image_data

print(cv2.__version__)


def classify_image(im_b64, model):
    countered_image = is_image_contoured(im_b64)
    # ML BEGINS
    if countered_image is not False:
        data = get_image_data(countered_image)
        # run the inference
        prediction = model.predict(data)
        for predict in prediction[0]:
            print(predict)
        label1, label2 = get_labels()
        return {label1: prediction[0].tolist()[0], label2: prediction[0].tolist()[1]}
    else:
        return {"result": "image has no 30 contours!"}