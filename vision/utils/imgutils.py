import os
import cv2
import numpy as np


class Image:
    def __init__(self):
        pass

    def bgr_to_grey(self, frame):
        c_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return c_frame

    def bgr_to_rgb(self, frame):
        c_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return c_frame

    def crop(self, img, cords):
        arr_img = np.asarray(img)
        crop_image = arr_img[
            cords[0]: cords[0] + cords[3], cords[1]: cords[2] + cords[4]
        ]
        return crop_image

    def read_img(self, img_path):
        file_exists = os.path.exists(img_path)
        if file_exists is True:
            self.frame = cv2.imread(img_path)
            return self.frame
        else:
            print("{0} does not exist".format(img_path))

    def read_video(self, video_path, webcam=False):
        if video_path is None:
            print("video path not supplied ")
        else:
            file_exists = os.path.exists(video_path)
            if file_exists is True:
                video = cv2.VideoCapture(video_path)
                return video
