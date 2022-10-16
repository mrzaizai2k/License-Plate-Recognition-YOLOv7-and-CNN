import time

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import transform_img
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized


def detect(model, image, device, imgsz=640, conf_thres=0.25,
           iou_thres=0.45, augment=False, classes=0, agnostic_nms=False):
    '''
    Find license Plate with YOLOv7
    :return:

    Pred:
        coordinates of LP
    im0:
        original image with LP plot
    '''
    # Initialize


    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    model.half()  # to FP16

    # Transform image to predict
    img, im0 = transform_img(image)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference

    t0 = time.time()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]
    t2 = time_synchronized()
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t3 = time_synchronized()
    final_pred = []

    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            final_pred.append(det)
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

        # Print time (inference + NMS)
        print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        print('Number of License Plate:', len(det))

        # cv2.imshow('Detected license plates', cv2.resize(im0, dsize=None, fx=0.5, fy=0.5))

    print(f'Done. ({time.time() - t0:.3f}s)')
    return final_pred[0].to(device='cpu').detach().numpy(), im0


def main():
    weights = 'LP_detect_yolov7_500img.pt'
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = attempt_load(weights, map_location=device)
    image_path = 'data/test/images/clip3_new_3.jpg'
    source_img = cv2.imread(image_path)
    cv2.imshow('input', cv2.resize(source_img, dsize=None, fx=0.5, fy=0.5))
    final_pred = detect(model, source_img,device, imgsz=640)
    print('final_pred', final_pred)

    # cv2.imshow('output', cv2.resize(source_img, dsize=None, fx=0.5, fy=0.5))
    # cv2.waitKey(0)


if __name__ == '__main__':
    main()
