import cv2
import numpy as np


def get_labels():
    with open('converted_keras/labels.txt') as f:
        lines = f.readlines()
        arr = []
        for line in lines:
            new_line = line.replace("0 ", "")
            new_line = new_line.replace("1 ", "")
            new_line = new_line.replace("\n", "")
            arr.append(new_line)
        return arr


def get_image_data(image):
    size = (224, 224)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    height, width, channels = image.shape
    scale_value = width / height
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center

    image = cv2.resize(image, size, fx=scale_value, fy=1, interpolation=cv2.INTER_NEAREST)
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    return data
