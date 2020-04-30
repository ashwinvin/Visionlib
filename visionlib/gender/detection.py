import cv2
import logging
import numpy as np
from visionlib.utils.webutils import web
from visionlib.face.detection import FDetector
from visionlib.utils.imgutils import Image


class GDetector:
    """
    This class contains all functions to detect gender of a given face
                . . .

    Methods:

        detect_gender():

            Used to detect gender from an face. Returns Predicted Gender and
            confidence.

    """

    def __init__(self):
        self.labels = ["male", "female"]
        self.mean = (78.4263377603, 87.7689143744, 114.895847746)
        self.f_detector = FDetector()
        self.web_util = web()
        self.f_detector.set_detector("hog")
        self.__set_detector()

    def __set_detector(self):
        model, proto = self.web_util.download_file('Gender')
        self.model = cv2.dnn.readNet(proto, model)

    def detect_gender(self, img=None, enable_gpu=False):
        '''This method is used to detect gender from an image.

        Args:
            img (numpy array)
                This argument must the output which similar to
                opencv's imread method's output.
            enable_gpu (bool) :
                Set to True if You want to use gpu for prediction.
        Returns:
            str :
                Returns the predicted gender.
            int :
                Returns the confidence for the predicted gender.
        '''
        if img != []:
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
