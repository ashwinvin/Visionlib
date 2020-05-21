import cv2
import logging
import unittest
from visionlib.utils.imgutils import Image
from visionlib.face.detection import FDetector
from visionlib.gender.detection import GDetector
from visionlib.keypoints.detection import KDetector


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cls is not TestData and cls.setUp is not TestData.setUp:
            orig_setUp = cls.setUp

            def setUpOverride(self, *args, **kwargs):
                TestData.setUp(self)
                return orig_setUp(self, *args, **kwargs)

            cls.setUp = setUpOverride

    def setUp(self):
        self.detectors = ["mtcnn", "haar", "hog", "dnn"]
        self.face_detector = FDetector()
        self.group_img_path = "tests/assets/face.jpg"
        self.test_img = cv2.imread(self.group_img_path)
        logging.basicConfig(level=logging.INFO)


class TestFaceDetection(TestData):
    def setUp(self):
        pass

    def test_face(self):
        logging.info("Starting Test for Face Detection")
        for detector in self.detectors:
            logging.info("Testing Detector : {}".format(detector))
            self.face_detector.set_detector(detector=detector)
            result = self.face_detector.detect_face(img=self.test_img)
            self.assertIsNotNone(result)


class TestKeypointDetection(TestData):
    def setUp(self):
        self.keypoint_detector = KDetector()
        self.detectors = ["dlib", "mtcnn"]

    def test_keypoint(self):
        logging.info("Starting Test for Facial Keypoint Detection")
        self.face_detector.set_detector("hog")
        for detector in self.detectors:
            logging.info("Testing Detector : {} ".format(detector))
            self.keypoint_detector.set_detector(detector)
            _, boxes, _ = self.face_detector.detect_face(self.test_img)
            result = self.keypoint_detector.detect_keypoints(
                img=self.test_img, rects=boxes
            )
            self.assertIsNotNone(result)


class TestGenderDetection(TestData):
    def setUp(self):
        self.image_util = Image()
        self.gender_detector = GDetector()

    def test_gender(self):
        logging.info("Starting Test for Gender Detection")
        self.face_detector.set_detector("hog")
        _, boxes, _ = self.face_detector.detect_face(self.test_img)
        logging.info("Testing Gender Detector")
        for box in boxes:
            c_img = self.image_util.crop(self.test_img, box)
            gender = self.gender_detector.detect_gender(c_img)
            self.assertIsNotNone(gender)

if __name__ == "__main__":
    unittest.main()
