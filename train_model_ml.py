import numpy as np

from check_contour import is_image_contoured
from read_file import get_labels, get_image_data


def train_model_func(model, im_b64, is_frozen):
    countered_image = is_image_contoured(im_b64)
    # ML BEGINS
    size = (224, 224)
    if countered_image is not False:
        data = np.asarray(get_image_data(countered_image))
        prediction = np.asarray([is_frozen])
        result = model.fit(data, prediction)
        print(result)
    return True
