import cv2
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from ..utils.imgutils import Image
from .haar_detector import HaarDetector
from .hog_detector import Hog_detector
from .mtcnn_detector import MTCNNDetector


class FDetector:
    """
    This class contains all functions to detect face in an image.
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
        """
        This method is used to set detector to be used to detect
        faces in an image.
        Args:
            detector: str
                    The detector to be used. Can be any of the following:
                    haar, hog, mtcnn
        """
        if detector == "haar":
            self.detector = self.haar
        elif detector == "hog":
            self.detector = self.hog
        elif detector == "mtcnn":
            self.detector = self.mtcnn

    def detect_face(self, img=None, show=False):
        """
        This method is used to detect face in an image.

        Args:
            img : cv2.imshow return output
                This argument must the output which similar to
                opencv's imread method's output.
            show (bool):  Set True to show image via cv2.imshow method.
        Returns:
            img (np.array) : Returns a numpy array of the image with bounding box.
                    ONLY given when show is set to True
            box (list) : Returns x, y, w, h coordinates of the detected face
                    Returns an empty list if no face is detected.

        """

        if img is not None:
            box = self.detector.detect(img=img)
            frame = None
            if box is not None:
                for face in box:
                    frame = cv2.rectangle(
                        img, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2
                    )

            if show is True:
                cv2.imshow("Visionlib", frame)
                cv2.waitKey(0)
                return [frame, box]
            else:
                return img if (frame is None) else [frame, box]

        else:
            raise Exception("No Arguments given")

    def vdetect_face(self, vid_path=None, show=False):
        '''
        This method is used to detect face in an video

        Args:
            vid_path : str
                Absolute Path to the video file.

            show (bool):  Set True to show image via cv2.imshow method.
        Yields:
            img (np.array) : Returns a numpy array of the image with bounding box.
                    ONLY given when show is set to True
            box (list) : Returns x, y, w, h coordinates of the detected face
                    Returns an empty list if no face is detected.
        '''
        vid = self.image_util.read_video(vid_path)
        while True:
            status, frame = vid.read()
            box = self.detector.detect(img=frame)
            if box is not None:
                for face in box:
                    frame = cv2.rectangle(
                        frame,
                        (face[0], face[1]),
                        (face[2], face[3]),
                        (0, 255, 0),
                        2,
                    )
                    yield [frame, box]
            if show is True:
                cv2.imshow("Visionlib", frame)
                yield [frame, box]
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
