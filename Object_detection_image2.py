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
import time
start_time = time.time()

# Name of the directory containing the object detection module we're using
# MODEL_NAME = 'inference_graph_inception'
# MODEL_NAME = 'inference_graph_resnet50'
# MODEL_NAME = 'inference_graph_resnet101'
MODEL_NAME = 'inference_graph_inception_resnet'
# MODEL_NAME = 'inference_graph_nas'
# DIR = 'testing_after_training/'
# DIR = 'pengujian_30/'
DIR = 'test_images/'
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
threshold = 50
threshold_limit = 50
interval = 10
jml_gbr_kelas = 30
detect_res = []
hasil = []
hasil_per_kelas = []
all_hasil = []
all_hasil_per_kelas = []
old_fp = 0
old_fn = 0
old_tp = 0
arr_have_correct_res = []

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
            min_score_thresh=threshold/100)
        print("--- %s seconds ---" % (time.time() - start_time))
        detect_res.append([image, boxes, classes, scores, category_index])
        DIR_RES = DIR+"temp/" + IMAGE_NAME
        # print(DIR_RES)
        # cv2.imwrite(DIR_RES, image)

# print(detect_res[0])
# print(detect_res[1])
while threshold <= threshold_limit:
    print(threshold)
    if threshold > 0:
        threshold_score = threshold / 100
    else:
        threshold_score = 0
    tp = 0
    fn = 0
    fp = 0
    j = 0
    arr_temp_correct = []
    # =====================================================================
    for IMAGE_NAME in listdir(DIR):
        # print("J:"+str(j))
        IMAGE_NAME = IMAGE_NAME.lower()
        if IMAGE_NAME == 'xml' or IMAGE_NAME == 'temp':
            # print(IMAGE_NAME)
            continue
        else:
            xml_path = get_xml_path(DIR_XML, IMAGE_NAME)
            true_classname = get_true_classname(IMAGE_NAME)
            image = detect_res[j][0]
            boxes = detect_res[j][1]
            classes = detect_res[j][2]
            scores = detect_res[j][3]
            category_index = detect_res[j][4]
            x21, x22, y21, y22 = get_gt_box(xml_path) # ground truth coordinate
            x21 = int(x21)
            x22 = int(x22)
            y21 = int(y21)
            y22 = int(y22)
            im_height = image.shape[0]
            im_width = image.shape[1]
            # print("WIDTH:"+str(im_width))
            # print("Height:" + str(im_height))
            false_box = 0
            true_box = 0
            index_iou = 0
            highest_iou = 0
            temp_tp = 0
            for i in range(0, len(classes[0])):
                if scores[0, i] >= threshold_score:
                    # print(IMAGE_NAME)
                    predict_classname = category_index.get(classes[0, i])
                    predict_classname = predict_classname['name']
                    # print(predict_classname)
                    if predict_classname == true_classname:
                        # Get IoU
                        true_box += 1
                        ymin, xmin, ymax, xmax = boxes[0, i]
                        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                      ymin * im_height, ymax * im_height)
                        x11, x12, y11, y12 = (int(round(left)), int(round(right)), int(round(top)), int(round(bottom)) )  # bounding box coordinate
                        iou = get_iou(x11, x12, y11, y12, x21, x22, y21, y22)
                        # print("IoU :"+str(iou))
                        if iou > highest_iou and iou > THRESHOLD_IOU:
                            index_iou = i
                            temp_tp = 1
                            highest_iou = iou
                            arr_temp_correct.append(IMAGE_NAME)

                    else:
                        false_box += 1

        fp = fp + (false_box + true_box - temp_tp)
        if temp_tp == 0:
            fn += 1
        else:
            tp += 1
        hasil.append([true_classname, tp, fp, fn])

        if (j + 1) % jml_gbr_kelas == 0:
            old_tp = tp - old_tp
            old_fp = fp - old_fp
            old_fn = fn - old_fn
            hasil_per_kelas.append([true_classname, old_tp, old_fp, old_fn])
            old_tp = tp
            old_fp = fp
            old_fn = fn
        j = j+1
    # =====================================================================

    # find precision and recall
    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    else:
        precision = 0
        recall = 0

    all_hasil.append(hasil)
    all_hasil_per_kelas.append([threshold, hasil_per_kelas])

    all_precision.append(precision)
    all_recall.append(recall)
    all_threshold.append(threshold)
    print("tp", tp)
    print("fp", fp)
    print("fn", fn)
    print("precision", precision)
    print("recall", recall)
    threshold += interval
    print(arr_temp_correct)

# for i in hasil_per_kelas:
#     print(i)
print(all_precision,all_recall,all_threshold)
status = write_to_xml2(MODEL_NAME, all_hasil_per_kelas)
# status = write_to_xml(MODEL_NAME, all_precision, all_recall, all_threshold)
# print(status)
