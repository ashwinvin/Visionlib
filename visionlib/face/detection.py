import cv2
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from ..utils.imgutils import Image
from .haar_detector import HaarDetector
from .hog_detector import Hog_detector
from .dnn_detector import DnnDetector
from .mtcnn_detector import MTCNNDetector


class FDetector:
    """This class contains all functions to detect face in an image.
                . . .

    Methods:

        detect_face():

            Used to detect face in an image. Returns the image with bounding
            boxes. Uses detector set by set_detector() method.

        vdetect_face():

            Used to detect face in a video. Yields the frame with bounding
            boxes. Uses detector set by set_detector() method.

        set_detector():

            Used to set detector to used by detect_face() method. If not set
            will use dnn detector as default.

    """

    def __init__(self):
        self.image_util = Image()
        self.set_detector()
        self.img = None

    def set_detector(self, detector="dnn"):
        """
        This method is used to set detector to be used to detect
        faces in an image.

        Args:
            detector (str)
                The detector to be used. Can be any of the following:
                haar, hog, mtcnn, dnn. Dnn will be used as default.
        """
        if detector == "haar":
            self.detector = HaarDetector()
        elif detector == "hog":
            self.detector = Hog_detector()
        elif detector == "mtcnn":
            self.detector = MTCNNDetector()
        elif detector == "dnn":
            self.detector = DnnDetector()

    def detect_face(self, img=None, show=False, enable_gpu=False):
        """This method is used to detect face in an image.

        Args:
            img (numpy array)
                This argument must the output which similar to
                opencv's imread method's output.

            show (bool)
                Set True to show image via cv2.imshow method.

        Returns:
            img (np.array)
                Returns a numpy array of the image with bounding box.

            box (list)
                Returns x, y, w, h coordinates of the detected face
                Returns an empty list if no face is detected.

            confidences (list)
                Returns the associated confidences for
                the detected face.

        """

        if img is not None:
            box, confidences = self.detector.detect(img=img)
            frame = None
            if box is not None:
                for face, conf in zip(box, confidences):
                    frame = cv2.rectangle(
                        img, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2
                    )

                    if None not in confidences:
                        # print(confidences)
                        confidences = [float(x) * 10 for x in confidences]

            if show is True:
                cv2.imshow("Visionlib", frame)
                cv2.waitKey(0)
                return [frame, box, conf]
            else:
                return [img, None, None] if (frame is None) else [frame, box, confidences]

        else:
            raise Exception("No Arguments given")

    def vdetect_face(self, vid_path=None, show=False, enable_gpu=False, url=False):
        """This method is used to detect face in an video

        Args:
            vid_path (str)
                Absolute Path to the video file.

            show (bool)
                Set True to show image via cv2.imshow method.
        Yields:
            img (np.array)
                Returns a numpy array of the image with bounding box.

                box (list)
                Returns x, y, w, h coordinates of the detected face
                Returns an empty list if no face is detected.

            confidences (list)
                Returns the associated confidences for
                the detected face.
        """
        if url:
            vid = cv2.VideoCapture(vid_path)
        else:
            vid = self.image_util.read_video(vid_path)
        while True:
            status, img = vid.read()
            box, confidences = self.detector.detect(img=img, enable_gpu=enable_gpu)
            frame = None
            conf = None
            if box is not None:
                frame = img
                for face, conf in zip(box, confidences):
                    frame = cv2.rectangle(
                        frame, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2
                    )

                if None not in confidences:
                    confidences = [float(x) * 10 for x in confidences]

            if show is True:
                if frame is not None:
                    cv2.imshow("Visionlib", frame)
                else:
                    cv2.imshow("Visionlib", img)
                yield [img, None, None] if (frame is None) else [
                    frame,
                    box,
                    confidences,
                ]
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            yield [img, None, None] if (frame is None) else [frame, box, confidences]
