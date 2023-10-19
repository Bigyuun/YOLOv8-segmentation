from ultralytics import YOLO
import torch
# print(torch.cuda.is_available())
import time
DEVICE='cuda'
#
# model = YOLO('yolov8n-seg.pt')
# model.to(DEVICE)


if __name__ == '__main__':
    model = YOLO('yolov8m-seg-custom-20231018.pt')
    model.to(DEVICE)
    model.train(data='config.yaml', batch=32, epochs=30, imgsz=640, plots=True)
