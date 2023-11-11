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
import pandas as pd
import csv

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
# dateStr = today.strftime("%d/%m/%Y")

# load model
yolo_LP_detect = torch.hub.load(
    'yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load(
    'yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

prev_frame_time = 0
new_frame_time = 0

vid = cv2.VideoCapture(0)
# vid = cv2.VideoCapture("./test_image/test.mp4")


def check_plate(plate):
    ub_list_plate = ["65A19777", "65A29999"]
    for i in ub_list_plate:
        if (i == plate):
            return 1
        else:
            return 0


plate_text = []
while (True):
    ret, frame = vid.read()

    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()

    for plate in list_plates:
        flag = 0
        x = int(plate[0])  # xmin
        y = int(plate[1])  # ymin
        w = int(plate[2] - plate[0])  # xmax - xmin
        h = int(plate[3] - plate[1])  # ymax - ymin
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (int(plate[0]), int(plate[1])), (int(
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
                        frame, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
                    cv2.putText(frame, "Detect: Car", (int(plate[0]), int(
                        plate[1]-40)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    # if (check_plate(list_read_plates) == 1):
                    #     cv2.putText(frame, "Status: Xe co quan UBND", (int(plate[0]), int(
                    #         plate[1]-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    # else:
                    #     cv2.putText(frame, "Status: Unknow Data", (int(plate[0]), int(
                    #         plate[1]-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.putText(frame, "Status: Xe co quan UBND", (int(plate[0]), int(
                        plate[1]-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.putText(frame, "Time: "+dt_string, (int(plate[0]), int(
                        plate[1]-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.putText(frame, "Plate: "+lp, (int(plate[0]), int(
                        plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    flag = 1
                    break
            if flag == 1:
                break
    # new_frame_time = time.time()
    # fps = 1/(new_frame_time-prev_frame_time)
    # prev_frame_time = new_frame_time
    # fps = int(fps)
    # cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX,
    #             3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('HE THONG NHAN DANG BIEN SO BANG HINH ANH', frame)
    plate_text += list_read_plates
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


def write_csv_plate(dataw, element):
    file_name = "test.csv"
    # header_fields = ["Plate Infomation", "Time", "Note"]
    csv_rows = [[dataw, dt_string, element]]
    with open(file_name, 'a') as csv_file:
        csv_writer = csv.writer(csv_file)
        # csv_writer.writerow(header_fields)
        csv_writer.writerows(csv_rows)
        # writer.writerow([dataw,"14:54 29-10-2023",element])

# from openpyxl import load_workbook


def write_excel_plate(dataw, element):
    marks_data = pd.DataFrame(
        [{"Plate Information": dataw, "Time": "15:14 29-10-2023", "Note": element}])
    # determining the name of the file
    file_name = r'D:\QuocHuy\Project\AI\License-Plate-Recognition\test.xlsx'
    # sheet_name="ubnd"
    # saving the excel
    # file_name.parent.mkdir(parents=True, exist_ok=True)
    # with pd.ExcelWriter(file_name, mode="a", if_sheet_exists='overlay') as writer:
    #     marks_data.to_excel(writer, sheet_name="ubnd")
    marks_data.to_excel(file_name)


def write_text_plate(dataw, time, note):
    f = open(".//plate_infomation.txt", "a")
    data = dataw+"  |  15:14 29-10-2023  |  " + note+"\n"
    f.write(data)
    f.close()


data = ""
# for plt in plate_text:
#     # print(check_plate(plt))
#     if (check_plate(plt) == 1):
#         #    write_text_plate(plt,"","Co quan UBND")
#         write_csv_plate(plt, "Co quan UBND")
#     else:
#         #    write_text_plate(plt,"","Khong co du lieu")
#         write_csv_plate(plt, "Khong co du lieu")


# print("============================")
# print("Danh sach UBND chap nhan luu thong")
# print(["65A19777", "65A29999"])
# print("Danh sach UBND camera ghi nhan duoc")
# print(plate_text)
# print('Export to Excel File successfully.')
