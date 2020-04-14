import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from mtcnn import MTCNN

import dlib
from ..utils.imgutils import Image


class FaceDetection:
    def __init__(self):
        self.dlib_hog_model = dlib.get_frontal_face_detector()
        self.mtcnn_model = MTCNN()
        self.haar_cascades = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.image_util = Image()
        self.x = []
        self.y = []
        self.w = []
        self.h = []

    def dlib_hog_detector(self, img_path):
        d_img = self.image_util.read_img(img_path=img_path)
        detected_img = self.dlib_hog_model(d_img, 1)
        for face in detected_img:
            self.x.append(face.left())
            self.y.append(face.top())
            self.w.append(face.right())
            self.h.append(face.bottom())
        return (self.x, self.y, self.w, self.h)

    def mtcnn_detector(self, img_path):
        m_img = self.image_util.read_img(img_path=img_path)
        detected_img = self.mtcnn_model.detect_faces(m_img)
        for face in detected_img:
            self.x.append(face["box"][0])
            self.y.append(face["box"][1])
            self.w.append(face["box"][2])
            self.h.append(face["box"][3])
        return (self.x, self.y, self.w, self.h)

    def haar_detector(self, img_path):
        h_img = self.image_util.read_img(img_path=img_path)
        grey_img = self.image_util.bgr_to_grey(h_img)
        detected_img = self.haar_cascades.detectMultiScale(grey_img, 1.3, 5)
        for (x, y, w, h) in detected_img:
            self.x.append(x)
            self.y.append(y)
            self.w.append(x + w)
            self.h.append(y + h)
        return (self.x, self.y, self.w, self.h)
