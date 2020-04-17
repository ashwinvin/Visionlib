import cv2
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from ..utils.imgutils import Image
from .haar_detector import HaarDetector
from .hog_detector import Hog_detector
from .mtcnn_detector import MTCNNDetector


class Detector:
    """
    This class contains all functions to detect face in an image.
                . . .

    Methods
    =======

        detect_face()
        ============
        Used to detect face in an image. Returns the image with bounding
        boxes. Uses detector set by set_detector() method.

        set_detector()
        ==============
        Used to set detector to used by detect_face() method. If not set
        will haar detector as default.

    """

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

    def detect_face(self, img_path=None, img=None, video=False, show=True):
        if img is not None:
            box = self.detector.detect(img=img)
            frame = None
            if box is not None:
                for face in box:
                    frame = cv2.rectangle(
                        img, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2
                    )
            return img if (frame is None) else frame

        elif img_path is not None and video is False:
            frame = self.image_util.read_img(img_path)
            box = self.detector.detect(img=frame)
            if box is not None:
                for face in box:
                    frame = cv2.rectangle(
                        frame, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2
                    )
            return img if (frame is None) else frame

        elif img_path is not None and video is True:
            vid = self.image_util.read_video(img_path)
            while True:
                status, frame = vid.read()
                box = self.detector.detect(img=frame)
                if box is not None:
                    for face in box:
                        frame = cv2.rectangle(
                            frame, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)
                if show is True:
                    cv2.imshow("Visionlib", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
# TODO return bounding box for images.
        else:
            raise Exception("No Arguments given")
