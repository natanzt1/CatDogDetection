######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from get_iou import *

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from os import listdir

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph_inception'
# MODEL_NAME = 'inference_graph_resnet50'
# MODEL_NAME = 'inference_graph_resnet101'
# MODEL_NAME = 'inference_graph_inception_resnet'
# MODEL_NAME = 'inference_graph_nas'
DIR = 'testing_after_training/'
DIR_DST = DIR+"result/"
DIR_XML = DIR+"xml/"
IMAGE_NAME = '2008_006599.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,DIR+IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 38

# Threshold for testing image
THRESHOLD_IOU = 0.5

# counter = 0;
# threshold = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
# for x in range(1, 100):
#     threshold = x/100
#     print("THRESHOLD :",x,"%")
#     all_tp = []
#     all_fp = []
#     all_fn = []
all_threshold = []
all_precision = []
all_recall = []
# threshold = 50
# threshold_limit = 100
threshold = 99.9
threshold_limit = 99.9
interval = 10
while threshold <= threshold_limit:
    # print(threshold)
    if threshold > 0:
        threshold_score = threshold/100
    else:
        threshold_score = 0
    tp = 0
    fn = 0
    fp = 0
    for IMAGE_NAME in listdir(DIR):
        IMAGE_NAME = IMAGE_NAME.lower()
        if IMAGE_NAME == 'xml' or IMAGE_NAME == 'temp':
            continue
        else:
            PATH_TO_IMAGE = os.path.join(CWD_PATH, DIR + IMAGE_NAME)
            image = cv2.imread(PATH_TO_IMAGE)
            image_expanded = np.expand_dims(image, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            # Draw the results of the detection (aka 'visualize the results')

            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=5,
                min_score_thresh=threshold_score)


            # print(scores)
            # print(category_index)
            xml_path = get_xml_path(DIR_XML, IMAGE_NAME)
            true_classname = get_true_classname(IMAGE_NAME)
            x21, x22, y21, y22 = get_gt_box(xml_path) # ground truth coordinate
            x21 = int(x21)
            x22 = int(x22)
            y21 = int(y21)
            y22 = int(y22)
            im_height = image.shape[0]
            im_width = image.shape[1]
            # print(xmin, xmax, ymin, ymax)
            false_box = 0
            true_box = 0
            index_iou = 0
            highest_iou = 0
            temp_tp = 0
            for i in range(0, len(classes[0])):
                if scores[0, i] >= threshold_score:
                    predict_classname = category_index.get(classes[0, i])
                    predict_classname = predict_classname['name']
                    # print(predict_classname)
                    # print(true_classname)
                    # Find TP, highest iou box
                    if predict_classname == true_classname:
                        # Get IoU
                        true_box += 1
                        (ymin, xmin, ymax, xmax) = boxes[0, i]
                        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                      ymin * im_height, ymax * im_height)
                        x11, x12, y11, y12 = ( int(round(left)), int(round(right)), int(round(top)), int(round(bottom)) )  # bounding box coordinate
                        iou = get_iou(x11, x12, y11, y12, x21, x22, y21, y22)
                        # print(iou)
                        if iou > highest_iou and iou > THRESHOLD_IOU:
                            index_iou = i
                            temp_tp = 1
                            highest_iou = iou

                    else:
                        false_box += 1
            # print(true_box)
            # print(false_box)
            # print(temp_tp)

            fp = fp + (false_box + true_box - temp_tp)
            if temp_tp == 0:
                fn += 1
            else:
                tp += 1
            # print("tp", tp)
            # print("fp", fp)
            # print("fn", fn)

                # if scores[0, i] > new_score and scores[0, i] > threshold:
                #     (ymin, xmin, ymax, xmax) = boxes[0, i]
                #     (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                #                                   ymin * im_height, ymax * im_height)
                #     index = i
                #     new_score = scores[0, i]
                #     print(IMAGE_NAME, category_index.get(classes[0, index]), scores[0, index], boxes[0, index])
                #
                #     predict_classname = category_index.get(classes[0, index])
                #     predict_classname = predict_classname['name']
                #     print(true_classname, predict_classname)
                #     x11, x12, y11, y12 = (int(round(left)), int(round(right)), int(round(top)), int(round(bottom))) # bounding box coordinate
                #     # print(x11, x12, y11, y12)
                #     # print(x21, x22, y21, y22)
                #     iou = get_iou(x11, x12, y11, y12, x21, x22, y21, y22)
                #     print(iou)


            # All the results have been drawn on image. Now display the image.
            # cv2.imwrite(DIR_DST+IMAGE_NAME,image)
            # cv2.imshow('Object detector', image)

            # Press any key to close the image
            cv2.waitKey(0)

            # Clean up
            cv2.destroyAllWindows()
            # break

            # counter += 1
            # if counter > 2:
            #     break


    # find precision and recall
    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    else:
        precision = 0
        recall = 0

    all_precision.append(precision)
    all_recall.append(recall)
    all_threshold.append(threshold)
    print("tp", tp)
    print("fp", fp)
    print("fn", fn)
    print("precision", precision)
    print("recall", recall)
    threshold += interval

# status = write_to_xml(MODEL_NAME, all_precision, all_recall, all_threshold)
# print(status)