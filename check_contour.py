import base64

import cv2  # OpenCV Library
import numpy as np


def is_image_contoured(im_b64):
    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    grayFrame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converting to gray image
    gaussianBlurFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)

    _, thresh = cv2.threshold(gaussianBlurFrame, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    crop_img = None
    max_counter_len = list()
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

        max_counter_len.append(len(approx))
        if len(approx) > 31:
            # cv2.putText(frame, "Star", coords, font, 1, colour, 1)
            crop_img = image[y: y + h, x: x + w]

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(grayFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    max_counter_len.sort()
    print(max_counter_len)
    if crop_img is not None:
        cv2.imwrite('testIMG.jpg', crop_img)
        return crop_img
    else:
        return False
