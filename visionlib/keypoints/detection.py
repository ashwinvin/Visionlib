from visionlib.keypoints.dlib_detector import DlibDetector
from visionlib.keypoints.mtcnn_detector import MTCNNDetector
import logging
import cv2

class KDetector:
    """This class contains functions for getting
        keypoints for a space
    Methods

        set_detector()
            Set the detector to detect keypoints

        detect_keypoints()
            Detect the keypoints in a face.

        draw_points()
            Draws the detected points in an image
    """
    def __init__(self):
        self.ddetector = DlibDetector()
        self.mdetector = MTCNNDetector()
        self.detector = self.ddetector

    def set_detector(self, detector):
        """This function is used to set the detector
            to detect keypoints.

        Args
            detector (str)
                Can be 'dlib' or 'mtcnn'.

        Raises
            Invalid Selection
                Raised when the selection is invalid
        """

        if detector == 'dlib':
            self.detector = self.ddetector
        elif detector == 'mtcnn':
            self.detector = self.mdetector
        else:
            raise Exception("Invalid Selection")

    def detect_keypoints(self, img, rects=None, enable_gpu=False):
        """This function is used to detect keypoints.

        Args
            img (numpy array)
                The image for detection.

            rects (list)
                The coordinates of face. (Not Needed for mtcnn detector)

            enable_gpu (bool):
                Set to True if you want to use gpu

        Returns
            list
                Returns detected keypoint for each face

            img
                Returned only when no face is there for detection.
        """
        if rects is None and self.ddetector == self.detector:
            logging.fatal("No face coordinates given aborting detection")
            return img
        else:
            points = self.detector.detect(img, rects, enable_gpu)
            return points

    def draw_points(self, img, points, show=False, color=(0, 50, 255)):
        """This function is used to draw the detected points
            into the image.

        Args
            img (numpy array)
                The image for drawing.

            points (list)
                The coordinates of the keypoints

            show (bool)
                Set to true if you want to display the image

            color (tuple)
                The color to display the points in.

        Returns
            numpy array
                The image with keypoints marked
        """
        for point in points:
            for (x, y) in point:
                img = cv2.circle(img, (x, y), 1, color, -1)

        if show:
            cv2.imshow("Keypoint Detector", img)
            cv2.waitKey(0)
        return img
