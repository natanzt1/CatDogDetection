import xmltodict
from xlwt import Workbook

def get_iou(x11, x12, y11, y12, x21, x22, y21, y22):
    # x11 = 4
    # x12 = 8
    # y11 = 4
    # y12 = 8
    #
    # x21 = 5
    # x22 = 8
    # y21 = 5
    # y22 = 8

    # Overlap Area
    x1 = max(x11, x21)
    y1 = max(y11, y21)
    x2 = min(x12, x22)
    y2 = min(y12, y22)

    overlap = max(0, (x2 - x1)) * max(0, (y2 - y1))
    # print("overlap", overlap)

    # IoU Area
    box_a = (x12-x11) * (y12-y11)
    box_b = (x22-x21) * (y22-y21)

    iou = overlap / (box_a + box_b - overlap)
    # print("Box", box_a, box_b)
    # print("iou", iou)
    return iou

def get_xml_path(DIR_XML, image):
    temp = str.split(image, ".")
    file = temp[0]
    filename = DIR_XML + file + '.xml'
    return filename

def get_gt_box(xml_path):
    file = open(xml_path, "r")
    doc = xmltodict.parse(file.read())
    # if 'kintamani' in xml_path:
    #     print("kintamani")
    # else:
    #     print("selain kintamani")

    bndbox = doc["annotation"]["object"]["bndbox"]
    xmin = bndbox["xmin"]
    xmax = bndbox["xmax"]
    ymin = bndbox["ymin"]
    ymax = bndbox["ymax"]
    return xmin, xmax, ymin, ymax

def get_true_classname(image):
    # print(image)
    temp = str.split(image, "_")
    classname = ''
    for x in range(0, len(temp)-1):
        if x == 0:
            classname = temp[x]
        else:
            classname = classname+"_"+temp[x]
    return classname



def write_to_xml(path, all_precision, all_recall, all_threshold):
    wb = Workbook()
    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1')

    for i in range(0, len(all_precision)):
        sheet1.write(i, 0, all_threshold[i])
        sheet1.write(i, 1, all_precision[i])
        sheet1.write(i, 2, all_recall[i])

    path = path+"/"+'testing1.xls'
    wb.save(path)
    return "Writting data to xml successfully"

def write_to_xml2(path, all_hasil_per_kelas):
    wb = Workbook()
    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1')
    k = 0
    for i in range(0, len(all_hasil_per_kelas)):
        threshold = all_hasil_per_kelas[i][0]
        isi = all_hasil_per_kelas[i][1]
        sheet1.write(k, 0, threshold)
        sheet1.write(k + 1, 0, "tp")
        sheet1.write(k + 2, 0, "fp")
        sheet1.write(k + 3, 0, "fn")
        for j in range(0, len(isi)):
            kelas = isi[j][0]
            tp = isi[j][1]
            fp = isi[j][2]
            fn = isi[j][3]
            sheet1.write(k, j + 1, kelas)
            sheet1.write(k + 1, j + 1, tp)
            sheet1.write(k + 2, j + 1, fp)
            sheet1.write(k + 3, j + 1, fn)
        k = k + 5

    path = path+"/"+'HASIL PER KELAS (30).xls'
    wb.save(path)
    return "Writting data to xml successfully"

