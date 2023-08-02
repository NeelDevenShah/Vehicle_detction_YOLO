# Conversion functions between RGB and other color systems.
import colorsys
# Recognize image file formats based on their first few bytes.
import imghdr
# OS routines for NT or Posix depending on what system we're on.
import os
import random
from keras import backend as K
import numpy as np
# PIL = Python Image Libray
from PIL import Image, ImageDraw, ImageFont


def read_classes(classes_path):
    with open(classes_path) as f:
        class_name = [c.strip() for c in class_name]
        return class_name


def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

    # The -1 in the reshape method tells NumPy to infer the row count from the length of the array and the remaining dimension. In this case, the remaining dimension is 2, so NumPy will calculate the row count as len(arr) // 2.

    # import numpy as np

    # arr = np.arange(10)

    # new_arr = arr.reshape(-1, 2)

    # print(new_arr)

    # OUTPUT

    # [[0 1]
    #  [2 3]
    #  [4 5]
    #  [6 7]
    #  [8 9]]


def generate_colors(class_name):
    hsv_tuples = [(x / len(class_name), 1., 1.)
                  for x in range(len(class_name))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(101010)  # Fixed seed for consisten colors across runs.
    random.seed(None)
    return colors


def scale_boxes(boxes, image_shape):
    """Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    # Stacks a list of rank R tensors into a rank R+1 tensor.
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes


def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(
        tuple(reversed(model_image_size)), Image.BICUBIC)
    # BICUBIC interpolation is a more advanced interpolation method that uses a polynomial of degree 3 to approximate the original image. This results in an even smoother image with fewer artifacts than bilinear interpolation, but it can be even slower.

    image_data = np.array(resized_image, 0)  # Add batch dimension
    image_data /= 255
    image_data = np.expand_dims(image_data, 0)  # Add batch normalization
    return image, image_data


def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    # font style is not written
    font = ImageFont.truetype(size=np.floor(
        3e-2*image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.3f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top+0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top+1])

        for i in range(thickness):
            draw.rectangle([left + i, top+i, right-i,
                           bottom-i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(
            text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
