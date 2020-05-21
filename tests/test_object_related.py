import cv2
import logging
import unittest
from visionlib.object.classifier.detection import CDetector
from visionlib.object.detection.detection import ODetection


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
        logging.basicConfig(level=logging.INFO)
        self.test_img_path = "assets/object.jpeg"
        self.test_img = cv2.imread(self.test_img_path)


class TestClassification(TestData):
    def setUp(self):
        self.classifiers = ["inception", "xception", "vgg"]
        self.c_detector = CDetector()

    def test_classifier(self):
        logging.info("Starting Test for Object Classifier")
        for classifier in self.classifiers:
            logging.info("Testing Classifier : {}".format(classifier))
            self.c_detector.set_detector(classifier)
            result = self.c_detector.predict(self.test_img)
            self.assertIsNotNone(result)

class TestObjectDetection(TestData):
    def setUp(self):
        self.detectors = ['Tiny-yolov3', 'Yolov3']
        self.object_detector = ODetection()

    def test_object(self):
        logging.info("Starting Test for Object Detection")
        for detector in self.detectors:
            logging.info("Testing Detector : {}".format(detector))
            self.object_detector.set_detector(detector)
            result = self.object_detector.detect_objects(self.test_img)
            self.assertIsNotNone(result)
