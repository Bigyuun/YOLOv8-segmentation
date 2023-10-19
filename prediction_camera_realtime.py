from ultralytics import YOLO

import REALSENSE_CV
import time
import os
import pandas as pd
import cv2
import pyrealsense2
import threading

# RealSense Camera init
# camera = REALSENSE_CV.RealSense()
# camera.camera_init()

# def camera_read():
#     while True:
#         try:
#             camera.read_once()
#
#         except KeyboardInterrupt:
#             print("Keyboard Interrupt: Exiting...")
#             thread_camera.join()
#
# thread_camera = threading.Thread(target=camera_read)
# thread_camera.start()
# results = camera.model.predict(source="0", show=True, conf=0.5, stream=True)  # accepts all formats

model = YOLO('model/yolov8n_custom_20231019.pt')
# model = YOLO('yolov8m-seg-custom-20231018.pt')
results = model.predict(source="output2.avi", show=True, conf=0.5, stream=True, save=True)  # accepts all formats
# results = model.predict(source="IMG_8947_720p.mp4", show=True, conf=0.5, stream=True, save=True)  # accepts all formats

# results = model(source=..., stream=True)  # generator of Results objects
for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs
    # print(boxes)
    # print(masks)
    # print(probs)




# loading YOLOv8-segmentation Model
# DEVICE='cuda'
#
# count=0
# while True:
#     st=time.time()
#     camera.model.predict(source="0", show=True, conf=0.5)  # accepts all formats
#     # camera.model.predict(source="0", show=True, conf=0.5, stream=True)  # accepts all formats
#     et=time.time()
#     print(count, " | ", et-st, " sec")
#     count += 1

