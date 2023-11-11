from datetime import datetime
from PIL import Image
import cv2
import torch
import math
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper

# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True, help='path to input image')
# args = ap.parse_args()
# print(args.image)

yolo_LP_detect = torch.hub.load(
    'yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load(
    'yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

# img = cv2.imread(args.image)
img = cv2.imread("test_image/4.jpg")
plates = yolo_LP_detect(img, size=1280)

plates = yolo_LP_detect(img, size=1280)
list_plates = plates.pandas().xyxy[0].values.tolist()
list_read_plates = set()

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

if len(list_plates) == 0:
    lp = helper.read_plate(yolo_license_plate, img)
    if lp != "unknown":
        cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (36, 255, 12), 2, 2)
        list_read_plates.add(lp)
else:
    for plate in list_plates:
        flag = 0
        x = int(plate[0])  # xmin
        y = int(plate[1])  # ymin
        w = int(plate[2] - plate[0])  # xmax - xmin
        h = int(plate[3] - plate[1])  # ymax - ymin
        crop_img = img[y:y+h, x:x+w]
        cv2.rectangle(img, (int(plate[0]), int(plate[1])), (int(
            plate[2]), int(plate[3])), color=(0, 0, 225), thickness=2)
        cv2.imwrite("crop.jpg", crop_img)
        rc_image = cv2.imread("crop.jpg")
        lp = ""
        for cc in range(0, 2):
            for ct in range(0, 2):
                lp = helper.read_plate(
                    yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    (text_width, text_height) = cv2.getTextSize(
                        lp, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)[0]
                    text_offset_x = int(plate[0])
                    text_offset_y = int(plate[1]-10)
                    # make the coords of the box with a small padding of two pixels
                    box_coords = ((text_offset_x, text_offset_y), (text_offset_x +
                                  text_width + 60, text_offset_y-30 - text_height - 2))
                    cv2.rectangle(
                        img, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
                    cv2.putText(img, "Detect: Car", (int(plate[0]), int(
                        plate[1]-40)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.putText(img, "Status: Unknow Data", (int(plate[0]), int(
                        plate[1]-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.putText(img, "Time: "+dt_string, (int(plate[0]), int(
                        plate[1]-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.putText(img, "Plate: "+lp, (int(plate[0]), int(
                        plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    flag = 1
                    break
            if flag == 1:
                break
cv2.imshow('HE THONG NHAN DANG BIEN SO BANG HINH ANH', img)
cv2.waitKey()
cv2.destroyAllWindows()

# python lp_image.py -i test_image/3.jpg