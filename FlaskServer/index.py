import csv
import pandas as pd
import function.helper as helper
import argparse
import time
from IPython.display import display
import function.utils_rotate as utils_rotate
import math
import torch
from PIL import Image
from datetime import datetime
from flask import Flask, render_template, Response, request
import cv2
import os
import sys
import numpy as np
from threading import Thread

# instatiate flask app
app = Flask(__name__, template_folder='./templates')


now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
# dateStr = today.strftime("%d/%m/%Y")

# load model
yolo_LP_detect = torch.hub.load(
    'yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load(
    'yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

vid = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0


def displayCamera():

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
        # plate_text += list_read_plates
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # vid.release()
    # cv2.destroyAllWindows()


def gen_frames():  # generate frame by frame from camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to capture image")
            break

        cv2.imwrite('demo.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('grey') == 'Grey':
            global grey
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if (face):
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':

            if (switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if (rec):
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(
                    str(now).replace(":", '')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out,])
                thread.start()
            elif (rec == False):
                out.release()

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

# vid.release()
# cv2.destroyAllWindows()
