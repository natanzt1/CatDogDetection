import sys
from decimal import Decimal
import cv2
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
import matplotlib.pyplot as plt
import xlrd

from GUI_function_detection_image import *
qtCreatorFile = "fix_gui.ui"  # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class Testing(QThread):
    def __init__(self, MainWindowClass):
        super(Testing, self).__init__(MainWindowClass)
        self.file_dir = MainWindowClass.file
        # QThread.__init__(MainWindowClass)
        # YolactProcessing.__init__()


    def run(self):
        self.parameter = {'dir': self.file_dir}
        print(self.parameter)
        self.start_process(self.update_frame,self.update_gui,self.parameter)
        print("hello")


class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.true_class = None
        self.image_path = None
        self.IMAGE_TEMP = 'C:/tensorflow1/models/research/object_detection/testing_after_training/temp/temp.jpg'
        # self.inputLog.setReadOnly(True)
        self.btn_open_image.clicked.connect(self.open_image)
        self.btn_run_detector.clicked.connect(self.run_detector)
        self.btn_generate_diagram.clicked.connect(self.show_diagram)

    def open_image(self):
        file_filter = "Image Files (*.png *.jpg )"
        options = QFileDialog.Options()
        path = "C://tensorflow1/models/research/object_detection/testing_after_training"
        filename = QFileDialog.getOpenFileName(self, "Open Image file",
                                               path, file_filter, options=options)
        if filename[0] != '':
            image_path = filename[0]
            self.image_path = image_path
            classname = self.get_classname(image_path)
            self.true_class = classname
            pixmap = QtGui.QPixmap(image_path)
            # img_width = pixmap.width()
            # img_height = pixmap.height()
            img_ori_label_height = self.image_ori.frameGeometry().height()
            pixmap_resized = pixmap.scaledToHeight(img_ori_label_height)
            self.image_ori.setPixmap(pixmap_resized)
            self.label_classname.setText(self.true_class)

        else:
            print("Pilih File!")

    def get_classname(self, image_path):
        temp = image_path.lower()
        classname = str.split(temp, '/')
        classname = classname[-1:]
        classname = classname[0]
        classname = str.split(classname)
        classname = classname[0]
        classname = str.split(classname, '_')
        return classname[0]

    def run_detector(self):
        # iou_threshold = self.edit_iou.toPlainText()
        iou_threshold = self.edit_confidence.toPlainText()
        conf_threshold = self.edit_confidence.toPlainText()
        self.radio_inception.isChecked()
        if self.radio_inception.isChecked():
            model = 'inference_graph_inception'
        elif self.radio_resnet50.isChecked():
            model = 'inference_graph_resnet50'
        elif self.radio_resnet101.isChecked():
            model = 'inference_graph_resnet101'
        elif self.radio_inception_resnet.isChecked():
            model = 'inference_graph_inception_resnet'
        else:
            model = 'inference_graph_nas'

        temp = detection_image(self.image_path, conf_threshold,
                        iou_threshold, model, self.true_class)
        predicted_classname = temp[0]
        predicted_score = temp[1]
        print(predicted_classname)
        print(predicted_score)
        result = ''
        for i in range(0, len(predicted_score)):
            if i == 0:
                result = predicted_classname[i]+':'+str(predicted_score[i])+'\n'
            else:
                result = result+predicted_classname[i]+':'+str(predicted_score[i])+'\n'
        print(result)
        pixmap = QtGui.QPixmap(self.IMAGE_TEMP)
        img_result_label_height = self.image_result.frameGeometry().height()
        pixmap_resized = pixmap.scaledToHeight(img_result_label_height)
        self.image_result.setPixmap(pixmap_resized)
        self.label_result.setText(result)

    def show_diagram(self):
        path = ['colab_inception_resnet_testing1.xls', 'colab_inception_testing1.xls', 'colab_resnet50_testing1.xls']
        line_label = ["Inception Resnet", "Inception", "Resnet50"]
        precision = []
        recall = []
        self.MplWidget.canvas.axes.clear()
        for i in range(0, 3):
            workbook = xlrd.open_workbook(path[i])
            sheet = workbook.sheet_by_index(0)
            precision.append([])
            recall.append([])
            for x in range(0, 98):
                precision[i].append(sheet.cell(x, 1).value)
                recall[i].append(sheet.cell(x, 2).value)
            # print(precision[i], recall[i])
            # print(line_label[i])
            self.MplWidget.canvas.axes.plot(precision[i], recall[i], label=line_label[i])
            # plt.plot(precision[i], recall[i], label=line_label[i])
        # plt.plot((precision[0], recall[0]), (precision[1], recall[1]))
        print("show")
        self.MplWidget.canvas.axes.set_title('Evaluation Result')
        self.MplWidget.canvas.axes.set_xlabel('Recall')
        self.MplWidget.canvas.axes.set_ylabel('Precision')
        self.MplWidget.canvas.axes.text(0.2, 0.4, '*Higher is better', {'color': 'r'})
        self.MplWidget.canvas.axes.legend()
        self.MplWidget.canvas.draw()
        print("success")

        # plt.show()

        # self.label_diagram.

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = MyWindow()
    ui.show()
    sys.exit(app.exec_())