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
MODEL_NAME = 'inference_graph_inception_jurnal'
# MODEL_NAME = 'inference_graph_resnet50_jurnal'
# MODEL_NAME = 'inference_graph_ssd_inception_jurnal'
# MODEL_NAME = 'inference_graph_nas'
# DIR = 'testing_after_training/'
# DIR = 'pengujian_30/'
# DIR = 'test_image_jurnal/'
DIR = 'test_image_jurnal_fix/'
# DIR = 'test_image_jurnal_fix_cropped/'
# DIR_DST = DIR+"result/"
DIR_XML = DIR+"xml/"
# DIR_TEST = DIR+"temp/"
# DIR_TEST = DIR+"temp2/"
DIR_TEST = DIR+"temp3/"
IMAGE_NAME = '2008_006599.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap_jurnal.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,DIR+IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 50

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
threshold_limit = 90
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


def get_class(filename):
    b = filename.split('.')
    c = b[0]
    d = c.split('_')

    for x in range(0, len(d) - 1):
        if x == 0:
            classname = d[x]
        else:
            classname = classname + '_' + d[x]
    classname = classname.split('/')
    classname = classname[len(classname)-1]
    return classname


all_far = []
all_frr = []
false_arr = []
no_result = []
all_classes = []
all_scores = []
all_imagename = []
for IMAGE_NAME in listdir(DIR):
    IMAGE_NAME = IMAGE_NAME.lower()
    if IMAGE_NAME == 'xml' or IMAGE_NAME == 'temp' or IMAGE_NAME == 'temp2' or IMAGE_NAME == 'temp3':
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
        # print("--- %s seconds ---" % (time.time() - start_time))
        detect_res.append([image, boxes, classes, scores, category_index])
        DIR_RES = DIR_TEST + IMAGE_NAME
        # print(IMAGE_NAME)
        classname = get_class(IMAGE_NAME)
        predict_classname = category_index.get(classes[0, 0])
        predict_classname = predict_classname['name']
        confidence = scores[0, 0]
        all_scores.append(confidence)
        all_imagename.append(IMAGE_NAME)
        all_classes.append(predict_classname)

        # if scores[0, 0] >= 0.5:
        #     predict_classname = category_index.get(classes[0, 0])
        #     predict_classname = predict_classname['name']
        #     if classname != predict_classname:
        #         print("a")
        #         # false_arr.append(IMAGE_NAME)
        #
        #     # print("class: "+ predict_classname)
        #     # print("score: "+ str(scores[0, 0]))
        #     # print("\n")
        # else:
        #     no_result.append(IMAGE_NAME)
        cv2.imwrite(DIR_RES, image)

# for temp_threshold in range(threshold, threshold_limit):
for temp_threshold in range(72, 73):
    false_name = []
    no_name = []
    false_arr = []
    no_result = []
    for x in range(0, len(all_imagename)):
        if all_scores[x] > temp_threshold/100:
            true_classname = get_class(all_imagename[x])
            predict_classname = all_classes[x]
            if true_classname != predict_classname:
                false_arr.append(IMAGE_NAME)
                false_name.append(all_imagename[x])
                # print('False predict:'+ all_imagename[x])
            # print("class: "+ predict_classname)
            # print("score: "+ str(scores[0, 0]))
            # print("\n")

        else:
            no_result.append(IMAGE_NAME)
            no_name.append(all_imagename[x])
            # print('No predict:'+all_imagename[x])

    all_far.append(len(false_arr))
    all_frr.append(len(no_result))
    all_threshold.append(temp_threshold)

print("FALSE RESULT")
for x in false_name:
    print(x)

print("\nNO RESULT")
for x in no_name:
    print(x)
# print(threshold)
# print("False classification: "+ str(len(false_arr)))
# # for x in false_arr:
# #     print(x)
# # print('\n')
# print("No object classified: " + str(len(no_result)))
# print("=========================")
# for x in no_result:
#     print(x)

# wb = Workbook()
# # add_sheet is used to create sheet.
# sheet1 = wb.add_sheet('Sheet 1')
# sheet1.write(0, 0, "threshold")
# sheet1.write(0, 1, "FAR")
# sheet1.write(0, 2, "FRR")
# for i in range(0, len(all_threshold)):
#     sheet1.write(i + 1, 0, all_threshold[i])
#     sheet1.write(i + 1, 1, all_far[i])
#     sheet1.write(i + 1, 2, all_frr[i])
#
# path = DIR_TEST+'1_FAR_FRR.xls'
# wb.save(path)
# print("Finished!")

# wb = Workbook()
# # add_sheet is used to create sheet.
# sheet1 = wb.add_sheet('Sheet 1')
#
# for i in range(0, len(detect_res)):
#     # threshold = all_hasil_per_kelas[i][0]
#     # isi = all_hasil_per_kelas[i][1]
#     for j in range(0, len(detect_res[i])):
#         sheet1.write(i, j, detect_res[i])
#
# path = DIR+"temp/"+'HASIL PER KELAS.xls'
# wb.save(path)
# print("Finished!")

# print(detect_res[0])
# print(detect_res[1])
# while threshold <= threshold_limit:
#     print(threshold)
#     if threshold > 0:
#         threshold_score = threshold / 100
#     else:
#         threshold_score = 0
#     tp = 0
#     fn = 0
#     fp = 0
#     j = 0
#     arr_temp_correct = []
#     # =====================================================================
#     for IMAGE_NAME in listdir(DIR):
#         # print("J:"+str(j))
#         IMAGE_NAME = IMAGE_NAME.lower()
#         if IMAGE_NAME == 'xml' or IMAGE_NAME == 'temp':
#             # print(IMAGE_NAME)
#             continue
#         else:
#             xml_path = get_xml_path(DIR_XML, IMAGE_NAME)
#             true_classname = get_true_classname(IMAGE_NAME)
#             image = detect_res[j][0]
#             boxes = detect_res[j][1]
#             classes = detect_res[j][2]
#             scores = detect_res[j][3]
#             category_index = detect_res[j][4]
#             x21, x22, y21, y22 = get_gt_box(xml_path) # ground truth coordinate
#             x21 = int(x21)
#             x22 = int(x22)
#             y21 = int(y21)
#             y22 = int(y22)
#             im_height = image.shape[0]
#             im_width = image.shape[1]
#             # print("WIDTH:"+str(im_width))
#             # print("Height:" + str(im_height))
#             false_box = 0
#             true_box = 0
#             index_iou = 0
#             highest_iou = 0
#             temp_tp = 0
#             for i in range(0, len(classes[0])):
#                 if scores[0, i] >= threshold_score:
#                     # print(IMAGE_NAME)
#                     predict_classname = category_index.get(classes[0, i])
#                     predict_classname = predict_classname['name']
#                     # print(predict_classname)
#                     if predict_classname == true_classname:
#                         # Get IoU
#                         true_box += 1
#                         # ymin, xmin, ymax, xmax = boxes[0, i]
#                         # (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
#                         #                               ymin * im_height, ymax * im_height)
#                         # x11, x12, y11, y12 = (int(round(left)), int(round(right)), int(round(top)), int(round(bottom)) )  # bounding box coordinate
#                         # iou = get_iou(x11, x12, y11, y12, x21, x22, y21, y22)
#                         # # print("IoU :"+str(iou))
#                         # if iou > highest_iou and iou > THRESHOLD_IOU:
#                         #     index_iou = i
#                         #     temp_tp = 1
#                         #     highest_iou = iou
#                         #     arr_temp_correct.append(IMAGE_NAME)
#
#                     else:
#                         false_box += 1
#
#         fp = fp + (false_box + true_box - temp_tp)
#         if temp_tp == 0:
#             fn += 1
#         else:
#             tp += 1
#         hasil.append([true_classname, tp, fp, fn])
#
#         if (j + 1) % jml_gbr_kelas == 0:
#             old_tp = tp - old_tp
#             old_fp = fp - old_fp
#             old_fn = fn - old_fn
#             hasil_per_kelas.append([true_classname, old_tp, old_fp, old_fn])
#             old_tp = tp
#             old_fp = fp
#             old_fn = fn
#         j = j+1
#     # =====================================================================
#
#     # find precision and recall
#     if tp > 0:
#         precision = tp / (tp + fp)
#         recall = tp / (tp + fn)
#     else:
#         precision = 0
#         recall = 0
#
#     # all_hasil.append(hasil)
#     # all_hasil_per_kelas.append([threshold, hasil_per_kelas])
#     #
#     # all_precision.append(precision)
#     # all_recall.append(recall)
#     # all_threshold.append(threshold)
#     # print("tp", tp)
#     # print("fp", fp)
#     # print("fn", fn)
#     # print("precision", precision)
#     # print("recall", recall)
#     # threshold += interval
#     # print(arr_temp_correct)

# for i in hasil_per_kelas:
#     print(i)
# print(all_precision,all_recall,all_threshold)
# status = write_to_xml2(MODEL_NAME, all_hasil_per_kelas)
# status = write_to_xml(MODEL_NAME, all_precision, all_recall, all_threshold)
# print(status)
