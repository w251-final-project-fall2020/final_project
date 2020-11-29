import sys
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from datetime import datetime

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import paho.mqtt.client as mqtt

LOCAL_MQTT_HOST='mosquitto'
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC='weight_detection'

REMOTE_MQTT_HOST='ec2-3-89-113-213.compute-1.amazonaws.com'
REMOTE_MQTT_PORT=1883
REMOTE_MQTT_TOPIC='food_detector_cloud'

source = '0'
device = select_device('')
half = device.type != 'cpu'

model = None
dataset = None
names = None

DELIMITER = b';;;;;;;;;;'

np.set_printoptions(threshold=sys.maxsize)

def initialize():
    global model, dataset, names

    weights, view_img, imgsz = opt.weights, opt.view_img, opt.img_size
       
    # Initialize
    set_logging()

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    view_img = True
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

def detect(weight, save_img=False):
    
    weights, view_img, imgsz = opt.weights, opt.view_img, opt.img_size

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    path, img, im0s, vid_cap = next(dataset)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Total detections
    num_items = len(pred)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()

        #save_path = str(save_dir / p.name)
        #txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')

        save_timestamp = str(datetime.now())
        
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Cut out image and send msg
            for *xyxy, conf, cls in reversed(det):

                label = names[int(cls)]
                confidence = '%.2f' % (conf)

                print("%s detected with confidence %s" % (label, confidence))

                x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                crop_img = im0[y1:y2, x1:x2]

                #percent by which the image is resized
                scale_percent = 20

                #calculate the 50 percent of original dimensions
                width = int(crop_img.shape[1] * scale_percent / 100)
                height = int(crop_img.shape[0] * scale_percent / 100)

                # dsize
                dsize = (width, height)

                # resize image
                output = cv2.resize(crop_img, dsize)

                rc, png = cv2.imencode('.png', output)
                image_bytes = png.tobytes()

                save_detected_image(
                    image_bytes,
                    save_timestamp,
                    str(i), str(num_items), 
                    label,
                    confidence, 
                    weight
                )

            # # Write results
            # for *xyxy, conf, cls in reversed(det):
            #     if save_txt:  # Write to file
            #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #         line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
            #         with open(txt_path + '.txt', 'a') as f:
            #             f.write(('%g ' * len(line)).rstrip() % line + '\n')

            #     if save_img or view_img:  # Add bbox to image
            #         label = '%s %.2f' % (names[int(cls)], conf)
            #         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, t2 - t1))

        # # Stream results
        # if view_img:
        #     cv2.imshow(p, im0)
        #     if cv2.waitKey(1) == ord('q'):  # q to quit
        #         raise StopIteration

        # # Save results (image with detections)
        # if save_img:
        #     if dataset.mode == 'images':
        #         cv2.imwrite(save_path, im0)
        #     else:
        #         if vid_path != save_path:  # new video
        #             vid_path = save_path
        #             if isinstance(vid_writer, cv2.VideoWriter):
        #                 vid_writer.release()  # release previous video writer

        #             fourcc = 'mp4v'  # output video codec
        #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        #         vid_writer.write(im0)

    # if save_txt or save_img:
    #     print('Results saved to %s' % save_dir)

    print('Done. (%.3fs)' % (time.time() - t0))


def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC)

def save_detected_image(image, save_timestamp, index, num_items, label, confidence, weight):
    
    msg = DELIMITER.join([
        image, 
        save_timestamp.encode('utf-8'), 
        index.encode('utf-8'), 
        num_items.encode('utf-8'), 
        label.encode('utf-8'), 
        confidence.encode('utf-8'), 
        weight.encode('utf-8')
    ])

    print("message size: ", len(msg))

    try:
        remote_mqttclient = mqtt.Client()
        remote_mqttclient.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)
        ret = remote_mqttclient.publish(REMOTE_MQTT_TOPIC, payload=msg, qos=0, retain=False)
    except:
        print("remote mqtt message sending failed\n")

def on_message(client, userdata, msg):
    try:
        print("message received!")
        # if we wanted to re-publish this message, something like this should work

        weight = msg.payload.decode('utf-8')

        with torch.no_grad():
            detect(weight)

    except:
        print("Unexpected error:", sys.exc_info()[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    initialize()

    local_mqttclient = mqtt.Client()
    local_mqttclient.on_connect = on_connect_local
    local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
    local_mqttclient.on_message = on_message

    # go into a loop
    local_mqttclient.loop_forever()

   
