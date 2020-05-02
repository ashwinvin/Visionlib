from visionlib.utils.webutils import web
import numpy as np
import logging
import dlib
import bz2
import cv2
import os

class DlibDetector:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.model = None
        self.web_util = web()
        self.__set_detector()

    def __set_detector(self):
        com_model = self.web_util.download_file("Keypoint")
        model_abs = os.path.split(str(com_model[0]))
        model_path = (
            model_abs[0] + os.path.sep + "shape_predictor_68_face_landmarks.dat"
        )
        if os.path.exists(model_path):
            self.model = dlib.shape_predictor(model_path)

        else:
            logging.info("Decompressing model")

            with bz2.open(com_model[0], "rb") as ifile:
                uncom = ifile.read()

            with open(model_path, "wb") as wfile:
                wfile.write(uncom)
                wfile.close()
            self.model = dlib.shape_predictor(model_path)

    def shape_to_np(self, detected, dtype="int"):
        coords = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            coords[i] = (detected.part(i).x, detected.part(i).y)
        return coords

    def detect(self, img, rects, enable_gpu=False):
        if enable_gpu:
            if dlib.cuda.get_num_devices() != 1:
                dlib.DLIB_USE_CUDA = True
            else:
                logging.fatal("No gpu is detected")
        rects2, points = [], []
        for rect in rects:
            rects2.append(dlib.rectangle(rect[0], rect[1], rect[2], rect[3]))
        for i, rect in enumerate(rects2):
            point = self.model(img, rect)
            point = self.shape_to_np(point)
            points.append(point)
        return points
