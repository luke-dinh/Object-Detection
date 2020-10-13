import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def swap_xy(boxes):
    return tf.stack([boxes[:,1], boxes[:,0], boxes[:,3], boxes[:,2]], axis = -1)

def convert_to_xywh(boxes):
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1)

def convert_to_corners(boxes):
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1)

def iou(boxes1, boxes2):
    corners_1 = convert_to_corners(boxes1)
    corners_2 = convert_to_corners(boxes2)
    lu = tf.maximum(corners_1[:,None,:2], corners_2[:,:2])
    rd = tf.minimum(corners_1[:,None,2:], corners_2[:,2:])
    intersection = tf.maximum(0.0, rd-lu)
    intersection_area = intersection[:,:,0] * intersection[:,:,1]
    area_1 = boxes1[:,2] * boxes1[:,3]
    area_2 = boxes2[:,2] * boxes2[:,3]
    union = tf.maximum(area_1[:,None] + area_2 - intersection_area, 1e-8)

    return tf.clip_by_value(intersection_area/union, 0.0,1.0)

def visualize_result(image, boxes, classes, scores, figsize = [8,8], color = [0,1,0], linewidth =1):

    image = np.array(image, dtype = np.uint8)
    plt.figure(figsize = figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()

    for box, cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 -x1, y2 - y1
        patch = plt.Rectangle([x1,y1], w, h, fill= False, edgecolor= color, linewidth= linewidth)
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
        plt.show()
        return ax

class anchorbox:

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1/3, 2/3]]

        self.num_of_anchors = len(self.aspect_ratios) * len(self.scales)

        self.strides = [2 ** i for i in range(3,8)]
        self.areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self.anchor_dims = self.compute_dims

    def compute_dims(self):
        anchor_dims_all = []
        for area in self.areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = np.sqrt(area/ratio)
                anchor_width = area/anchor_height
                dims = np.reshape(np.stack([anchor_width, anchor_height], axis = -1), [1,1,2])
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(np.stack(anchor_dims, axis = -2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):

        rx = np.range(feature_width, dtype = np.float) + 0.5
        ry = np.range(feature_height, dtype = np.float) + 0.5
        centers = np.stack(np.meshgrid(rx,ry), axis= -1) * self._strides[level - 3]
        centers = np.expand_dims(centers, axis=-2)
        centers = np.tile(centers, [1, 1, self._num_anchors, 1])
        dims = np.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = np.concatenate([centers, dims], axis=-1)
        return np.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                np.ceil(image_height / 2 ** i),
                np.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return np.concatenate(anchors, axis=0)




