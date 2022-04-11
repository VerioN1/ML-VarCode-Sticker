import base64

import cv2  # OpenCV Library
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

print(cv2.__version__)


def classify_image(im_b64):
    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    grayFrame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converting to gray image
    gaussianBlurFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)

    _, thresh = cv2.threshold(gaussianBlurFrame, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    temp_w = ''
    temp_x = ''
    temp_y = ''
    temp_h = ''
    scale_value = None
    size = (224, 224)
    # Iterating through each contour to retrieve coordinates of each shape
    for i, contour in enumerate(contours):
        if i == 0:
            continue
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(gaussianBlurFrame, contour, 0, (255, 0, 0), 0)

        # Retrieving coordinates of the contour so that we can put text over the shape.
        x, y, w, h = cv2.boundingRect(approx)
        x_mid = int(x + (w / 3))  # This is an estimation of where the middle of the shape is in terms of the x-axis.
        y_mid = int(y + (h / 1.5))  # This is an estimation of where the middle of the shape is in terms of the y-axis.

        # Setting some variables which will be used to display text on the final image
        coords = (x_mid, y_mid)
        colour = (0, 0, 255)
        font = cv2.FONT_HERSHEY_DUPLEX

        if len(approx) > 35:
            # cv2.putText(frame, "Star", coords, font, 1, colour, 1)
            crop_img = image[y: y + h, x: x + w]

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(grayFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    model = load_model('./converted_keras/keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = crop_img
    height, width, channels = crop_img.shape
    scale_value = width / height
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center

    image = cv2.resize(image, size, fx=scale_value, fy=1, interpolation=cv2.INTER_NEAREST)
    # image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    for predict in prediction[0]:
        print(predict)
    return {"white": prediction[0].tolist()[0], "green": prediction[0].tolist()[1]}
