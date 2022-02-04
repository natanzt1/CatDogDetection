import sys
from decimal import Decimal
import cv2
import xlrd
import tensorflow as tf
from GUI_function_detection_image import *


from os import listdir
path = "C:/Users/Ritarion/Documents/Laravel/CatDogDetection/public/detection/raw/"
filename = listdir(path) # * means all if need specific format then *.csv
# print(filename)
path = path + filename[-1]
# print(path)
temp = detection_image_android(path)
predicted_classname = temp[0]
predicted_score = temp[1]
result = 0
for i in range(0, len(predicted_score)):
    if i == 0:
        result = predicted_classname[i] + ':' + str(predicted_score[i]) + '\n'
    else:
        result = result + predicted_classname[i] + ' : ' + str(predicted_score[i]) + '\n'
if result is not 0:
    print(result)
else:
    print("Tidak ada objek terdeteksi")