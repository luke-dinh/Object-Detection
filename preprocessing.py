import tensorflow as tf
import numpy as np
from Bounding_Box import *

def flip_horizontal(image, boxes):

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = np.stack(
            [1 - boxes[:,2], boxes[:,1], 1 - boxes[:,0], boxes[:,3]], axis=-1)
    return image, boxes

def resize_and_pad(image, min = 800.0, max = 1330.0, jitter = [640,1024], stride = 128.0):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride """

    image_shape = tf.cast(tf.shape(image)[:2], dtype = np.float)
    if jitter is not None:
        min = np.random.uniform((), jitter[0], jitter[1], dtype = np.float)
    ratio = min/tf.reduce_min(image_shape)

    if ratio * tf.reduce_max(image_shape) > max:
        ratio = max/tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype = np.int32))
    padded_image = tf.cast(np.ceil(image_shape/stride) * stride, dtype = np.int32)
    image = tf.image.pad_to_bounding_box(image, 0,0, padded_image[0], padded_image[1])

    return image, ratio, image_shape

def preprocessing(sample):

    image = sample['image']

    bbox = swap_xy(sample["object"]["bbox"])

    class_id = np.cast(sample["object"]["label"])

    image_flip = flip_horizontal(image, bbox)

    image, image_shape, _ = resize_and_pad(image)

    bbox = np.stack([
        bbox[:,0] * image_shape[1],
        bbox[:,1] * image_shape[0],
        bbox[:,2] * image_shape[1], 
        bbox[:,3] * image_shape[0]],axis = -1)

    bbox = convert_to_xywh(bbox)

    return bbox, image, class_id




