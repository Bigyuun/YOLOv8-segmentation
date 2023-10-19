import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

class RealSense:
    '''
    YOLOv8-segmentation Model & RealSense Camera Demo
    '''
    def __init__(self):

        self.DEVICE = 'cuda'
        self.model = YOLO('yolov8m-seg-custom-20230927.pt')
        self.model.to(self.DEVICE)

        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.image_size = {'width': 640,
                           'height': 480}
        self.fps = 30
        self.profile = None
        self.depth_sensor = None
        self.depth_scale = None
        self.frame = None
        self.depth_frame = None
        self.color_frame = None
        self.frame_loss = 0
        self.depth_image = None
        self.color_image = None
        # self.depth_image = np.zeros((self.image_size['width'], self.image_size['height']), dtype=int)
        # self.color_image = np.zeros((self.image_size['width'], self.image_size['height'], 3), dtype=int)
        self.depth_cm = None

    def camera_init(self):
        self.cfg.enable_stream(rs.stream.color,
                               self.image_size['width'],
                               self.image_size['height'],
                               rs.format.bgr8,
                               self.fps)
        self.cfg.enable_stream(rs.stream.depth,
                               self.image_size['width'],
                               self.image_size['height'],
                               rs.format.z16,
                               self.fps)

        # Start steaming
        self.profile = self.pipe.start(self.cfg)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print("Depth Scale : {}".format(self.depth_scale))

        print('Camera Initializing finish')

    def read(self):
        while True:
            # wait for a coherent pair of frames : depth and color
            self.frame = self.pipe.wait_for_frames()
            self.depth_frame = self.frame.get_depth_frame()
            self.color_frame = self.frame.get_color_frame()
            if not self.depth_frame or not self.color_frame:
                self.frame_loss = self.frame_loss + 1
                print("frame loss : {}".format(self.frame_loss))
                continue

            self.depth_image = np.asanyarray(self.depth_frame.get_data())
            self.color_image = np.asanyarray(self.color_frame.get_data())
            self.depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.5),
                                         cv2.COLORMAP_JET)

            result = self.model.predict(source=self.color_image, conf=0.5, save=True)

            gray_image = cv2.cvtColor(self.color_image,
                                      cv2.COLOR_BGR2GRAY)

            cv2.imshow('rgb', self.color_image)
            cv2.imshow('depth', self.depth_cm)

            if cv2.waitKey(1) == ord('q'):
                self.pipe.stop()
                break

    def read_once(self):
        # wait for a coherent pair of frames : depth and color
        self.frame = self.pipe.wait_for_frames()
        self.depth_frame = self.frame.get_depth_frame()
        self.color_frame = self.frame.get_color_frame()
        if not self.depth_frame or not self.color_frame:
            self.frame_loss = self.frame_loss + 1
            print("frame loss : {}".format(self.frame_loss))

        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.color_image = np.asanyarray(self.color_frame.get_data())
        self.depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.5),
                                     cv2.COLORMAP_JET)

        gray_image = cv2.cvtColor(self.color_image,
                                  cv2.COLOR_BGR2GRAY)

        # cv2.imshow('rgb', self.color_image)
        # cv2.imshow('depth', self.depth_cm)

        if cv2.waitKey(1) == ord('q'):
            self.pipe.stop()

if __name__ == '__main__':
    camera = RealSense()
    camera.camera_init()
    camera.read()

