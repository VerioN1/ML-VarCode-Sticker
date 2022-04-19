import base64

import cv2  # OpenCV Library
from keras.models import load_model
import numpy as np

from check_contour import is_image_contoured
from read_file import get_labels

print(cv2.__version__)


def classify_image(im_b64):
    countered_image = is_image_contoured(im_b64)
    size = (224, 224)
    if countered_image is not False:
        model = load_model('./converted_keras/keras_model.h5')
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # Replace this with the path to your image
        image = countered_image
        height, width, channels = image.shape
        scale_value = width / height
        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center

        image = cv2.resize(image, size, fx=scale_value, fy=1, interpolation=cv2.INTER_NEAREST)
        # image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        for predict in prediction[0]:
            print(predict)
        label1, label2 = get_labels()
        return {label1: prediction[0].tolist()[0], label2: prediction[0].tolist()[1]}
    else:
        return {"result": "image has no 30 contours!"}