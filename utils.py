"""Some usefull functions"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def segmentation_plot(image, mask, seg_color=(255, 0, 0)):
    """Return instance of matplotlib figure with 3 images:
    image, mask, mask on image.

    :image (array [N, M, 3]): source image
    :mask (array [N, M]): source mask
    :return: instance of matplotlib.pyplot.figure
    """
    blend_image = image.copy()
    blend_image[mask == 1.] = (blend_image[mask == 1.] + seg_color) // 2

    fig = plt.figure(figsize=(20, 20))
    fig.add_subplot(1, 3, 1)
    plt.imshow(image)
    fig.add_subplot(1, 3, 2)
    plt.imshow(mask, cmap='Greys')
    fig.add_subplot(1, 3, 3)
    plt.imshow(blend_image)
    return plt

def resize_pad(image, height, width):
    """Resize image with saving aspect ratio,
    pad image with zeros for given shape

    :image (array [N, M, K] or [N, M]): source image
    :height (int): height of new image
    :width (int): width of new image

    :return (array [height, width, K] or [height, width]): new image
    """
    new_shape = (height, width, image.shape[2]) if len(image.shape) == 3 else (height, width)
    new_image = np.zeros(new_shape, dtype=image.dtype)
    _dx = height / image.shape[0]
    _dy = width / image.shape[1]
    scale = min(_dx, _dy)
    resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    start_row = (height - resized_image.shape[0]) // 2
    start_col = (width - resized_image.shape[1]) // 2
    new_image[start_row:start_row + resized_image.shape[0],
              start_col:start_col + resized_image.shape[1]] = resized_image
    return new_image
