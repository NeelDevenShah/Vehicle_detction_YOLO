import numpy as np
import xml.etree.ElementTree as ET
import os
from PIL import Image

# The data.npz file will be created in the images/ folder


def generate_npz_data_file(image_dir, bounding_boxes_dir):
    """Generating a .npz data file training of model

    Args:
        images_dir: The directory containing the images.
        bounding_boxes_dir: The directory containing the bounding boxes.

    Returns:
        A .npz data file containing the images and bounding boxes.
  """
    images = []
    boxes = []

    for filename in os.listdir(image_dir):
        image = Image.open(image_dir + '/' + filename)
        image_data = np.array(image)
        images.append(image)

    for filename in os.listdir(bounding_boxes_dir):
        tree = ET.parse(bounding_boxes_dir + '/' + filename)
        root = tree.getroot()

        boxes_array = []
        for object in root.findall('object'):
            bounding_box = object.find('bndbox')
            xmin = int(bounding_box.find('xmin').text)
            ymin = int(bounding_box.find('ymin').text)
            xmax = int(bounding_box.find('xmax').text)
            ymax = int(bounding_box.find('ymax').text)

            box = [xmin, ymin, xmax, ymax]
            boxes_array.append(box)

    data = {'images': images, 'boxes': boxes}

    np.savez_compressed(image_dir + '/data.npz', **data)

    print('Operation Successfull')


if __name__ == '__main__':
    generate_npz_data_file('data/images', 'data/bounding_boxes')
