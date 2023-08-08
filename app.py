from keras import backend as K
from keras.models import load_model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes
from yad2k.model.keras_yolo import yolo_head
from app_utils import yolo_eval
from os import listdir
from os.path import isfile, join
import os

# Create a session to start graph
sess = K.get_session()

class_names = read_classes("data/pretrained_model/classes.txt")
anchors = read_anchors("data/pretrained_model/yolo_anchors.txt")
image_shape = (720., 1280.)

# Read yolo model
yolo_model = load_model("data/pretrained_model/yolo.h5")

# Preprocessing on yolo_model.output
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

# Filter boxes on NMS and thresholding
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)


def predict(sess, image_file):
    """
    Run the graph stored in "sess" to predict boxes for "image_file". Print and plot the predictions.

    Arguments:
    sess -- tensorflow/keras session containing the YOLO graph
    image_file -- name of an image stored in the "image" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    """

    # Preprocess your image
    image, image_data = preprocess_image(
        "data/images/" + image_file, model_image_size=(608, 608))

    # Run the yolo model
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={
                                                  yolo_model.input: image_data, K.learning_phase(): 0})

    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("data/images", image_file), quality=90)

    return out_scores, out_boxes, out_classes


# Run predict for all files in images/directory ans save it to out/directory
all_outputs = [predict(sess, f) for f in listdir(
    'data/images/') if isfile(join('data/images/', f))]
