# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from get_iou import *
from utils import label_map_util
from utils import visualization_utils as vis_util
import time
from os import listdir

sys.path.append("..")


def detection_image(path, confidence, iou, model, classname):
    model_name = model
    # print(path)

    DIR = 'testing_after_training/'
    DIR_DST = DIR + "temp/"
    DIR_XML = DIR + "xml/"

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, model_name, 'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

    # Path to image
    PATH_TO_IMAGE = path

    # Number of classes the object detector can identify
    NUM_CLASSES = 38

    # Threshold for testing image
    THRESHOLD_IOU = float(iou)

    THRESHOLD_SCORE = (float(confidence)) / 100

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
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

    # IMAGE_NAME = IMAGE_NAME.lower()
    image = cv2.imread(path)
    image_expanded = np.expand_dims(image, axis=0)

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
        min_score_thresh=THRESHOLD_SCORE)
    image_name = str.split(path, '/')
    image_name = image_name[-1:]
    image_name = image_name[0].lower()
    xml_path = get_xml_path(DIR_XML, image_name)
    true_classname = classname
    x21, x22, y21, y22 = get_gt_box(xml_path)  # ground truth coordinate
    x21 = int(x21)
    x22 = int(x22)
    y21 = int(y21)
    y22 = int(y22)
    im_height = image.shape[0]
    im_width = image.shape[1]
    false_box = 0
    true_box = 0
    highest_iou = 0
    temp_tp = 0
    fp = 0
    fn = 0
    tp = 0
    all_predicted_classname = []
    all_predicted_score = []
    for i in range(0, len(classes[0])):
        if scores[0, i] >= THRESHOLD_SCORE:
            predict_classname = category_index.get(classes[0, i])
            predict_classname = predict_classname['name']
            all_predicted_classname.append(predict_classname)
            all_predicted_score.append(scores[0, i])
            # Find TP, highest iou box
            if predict_classname == true_classname:
                # Get IoU
                true_box += 1
                (ymin, xmin, ymax, xmax) = boxes[0, i]
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)
                x11, x12, y11, y12 = (
                int(round(left)), int(round(right)), int(round(top)), int(round(bottom)))  # bounding box coordinate
                iou = get_iou(x11, x12, y11, y12, x21, x22, y21, y22)
                # print(iou)
                if iou > highest_iou and iou > THRESHOLD_IOU:
                    index_iou = i
                    temp_tp = 1
                    highest_iou = iou

            else:
                false_box += 1

    fp = fp + (false_box + true_box - temp_tp)
    if temp_tp == 0:
        fn += 1
    else:
        tp += 1

    # All the results have been drawn on image. Now display the image.
    cv2.imwrite(DIR_DST + 'temp.jpg', image)

    # find precision and recall
    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    else:
        precision = 0
        recall = 0

    return (all_predicted_classname, all_predicted_score)
    # print("tp", tp)
    # print("fp", fp)
    # print("fn", fn)
    # print("precision", precision)
    # print("recall", recall)
    #


def detection_image_android(path):
    # print("ITU PATHNYA")
    # print(path)

    confidence = 50
    iou = 0.5
    model_name = 'inference_graph_inception'

    # Grab path to current working directory
    image_name = str.split(path, '/')
    # print(image_name)
    image_name = image_name[-1]
    # print(image_name)
    CWD_PATH = "C:/tensorflow1/models/research/object_detection"
    DIR_RES = "C:/Users/Ritarion/Documents/Laravel/CatDogDetection/public/detection/result/"+image_name
    # print(DIR_RES)

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, model_name, 'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

    # Path to image
    PATH_TO_IMAGE = path

    # Number of classes the object detector can identify
    NUM_CLASSES = 38

    # Threshold for testing image
    THRESHOLD_IOU = float(iou)

    THRESHOLD_SCORE = (float(confidence)) / 100

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
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

    # IMAGE_NAME = IMAGE_NAME.lower()
    image = cv2.imread(path)
    # cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    # image = cv2.imread(path)
    image_expanded = np.expand_dims(image, axis=0)

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
        min_score_thresh=THRESHOLD_SCORE)
    all_predicted_classname = []
    all_predicted_score = []
    for i in range(0, len(classes[0])):
        if scores[0, i] >= THRESHOLD_SCORE:
            predict_classname = category_index.get(classes[0, i])
            predict_classname = predict_classname['name']
            all_predicted_classname.append(predict_classname)
            all_predicted_score.append(scores[0, i])

    # All the results have been drawn on image. Now display the image.
    cv2.imwrite(DIR_RES, image)
    # print("B")
    time.sleep(2)
    return (all_predicted_classname, all_predicted_score)
