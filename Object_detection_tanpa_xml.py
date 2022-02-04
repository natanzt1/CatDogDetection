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
# DIR = 'pengujian/'
DIR = 'update/'
DIR_DST = DIR+"result/"
# DIR_XML = DIR+"xml/"

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

all_threshold = []
all_precision = []
all_recall = []
# threshold = 50
# threshold_limit = 100
threshold = 0.70
threshold_limit = 51
interval = 1
jml_gbr_kelas = 10
detect_res = []
hasil = []
hasil_per_kelas = []
all_hasil = []
all_hasil_per_kelas = []
old_fp = 0
old_fn = 0
old_tp = 0

for IMAGE_NAME in listdir(DIR):
    IMAGE_NAME = IMAGE_NAME.lower()
    if IMAGE_NAME == 'ganti.txt' or IMAGE_NAME == 'result' or IMAGE_NAME == 'hasil.txt':
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
            min_score_thresh=threshold)
        detect_res.append([image, boxes, classes, scores, category_index])
        # DIR_RES = DIR+"temp/" + IMAGE_NAME
        # print(DIR_RES)
        DIR_RES = DIR_DST + IMAGE_NAME
        print(DIR_RES)
        cv2.imwrite(DIR_RES, image)