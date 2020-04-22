import cv2
import logging
import numpy as np
from visionlib.face.detection import FDetector
from visionlib.utils.imgutils import Image


class GDetector:
    def __init__(self):
        self.labels = ["male", "female"]
        self.mean = (78.4263377603, 87.7689143744, 114.895847746)
        self.proto = "/home/ashwin/Vision/visionlib/models/gender_deploy.prototxt"
        self.model_path = "/home/ashwin/Vision/visionlib/models/gender_net.caffemodel"
        self.model = cv2.dnn.readNetFromCaffe(self.proto, self.model_path)
        self.f_detector = FDetector()
        self.f_detector.set_detector("hog")
        self.im_utils = Image()

    def detect_gender(self, img=None, enable_gpu=False):
        if img is not None:
            blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), self.mean, swapRB=False)

            if enable_gpu:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            self.model.setInput(blob)
            preds = self.model.forward()
            preds = preds[0]
            b_pred = self.labels[np.argmax(preds)]
            b_confidence = preds[np.argmax(preds)]
            return (b_pred, b_confidence)
        else:
            print("img is none")
