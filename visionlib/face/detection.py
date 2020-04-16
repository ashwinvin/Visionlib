import cv2
from ..utils.imgutils import Image
from .haar_detector import HaarDetector
from .hog_detector import Hog_detector
from .mtcnn_detector import MTCNNDetector


class Detector:
    def __init__(self):
        self.image_util = Image()
        self.hog = Hog_detector()
        self.haar = HaarDetector()
        self.mtcnn = MTCNNDetector()
        self.detector = self.haar
        self.img = None

    def set_detector(self, detector):
        if detector == "haar":
            self.detector = self.haar
        elif detector == "hog":
            self.detector = self.hog
        elif detector == "mtcnn":
            self.detector = self.mtcnn

    def detect_face(self, img_path=None, webcam=False, video_path=None):
        if webcam is not False:
            web_vid = cv2.VideoCapture(0)
            while True:
                status, frame = web_vid.read()
                box = self.detector.detect(img=frame)
                frame = cv2.rectangle(
                    frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2
                )
                cv2.imshow("Smart Eye", frame)
                cv2.waitKey(0)
        elif video_path is not None:
            video = self.image_util.read_video(video_path)
            while True:
                status, frame = video.read()
                box = self.detector.detect(img=frame)
                frame = cv2.rectangle(
                    frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2
                )
                cv2.imshow("Smart Eye", frame)
                cv2.waitKey(0)
        elif img_path is not None:
            frame = self.image_util.read_img(img_path)
            box = self.detector.detect(img=frame)
            for face in box:
                frame = cv2.rectangle(
                    frame, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2
                )
            return box
        else:
            raise Exception("No Arguments given")
