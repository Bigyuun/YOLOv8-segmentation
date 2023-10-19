from ultralytics import YOLO
import torch
# print(torch.cuda.is_available())
import time
DEVICE='cuda'
#
# model = YOLO('yolov8n-seg.pt')
# model.to(DEVICE)


if __name__ == '__main__':
    model = YOLO('yolov8m-seg-custom-20230927.pt')
    model.to(DEVICE)
    count = 0
    st = time.time()

    for i in range(1):
        results = model.predict('20230926-034253_117_png.rf.089e710a13c537c67345278801f16237.jpg', conf=0.5, save=True)
        results = model.predict('20230926-034254_130_png.rf.19cb8747acf4bd777e0696f88f8c423a.jpg', conf=0.5, save=True)
        results = model.predict('20230926-034255_143_png.rf.2a0b72507c75b5ffdf2da18d8d87e0ff.jpg', conf=0.5, save=True)
        # results = model.predict('20230927-144957_1940.png', conf=0.25, save=True)
        et=time.time()
        # print(et-st)
        count += 1
        print(count)

    print(et-st)