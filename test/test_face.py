import unittest
from visionlib.face.detection import Detector


class TestDetection(unittest.TestCase):
    def test_hog(self):
        detect = Detector()
        img = "/home/ashwin/Vision/test/assets/face.jpeg"
        set_detector = detect.set_detector(detector="hog")
        result = detect.detect_face(img_path=img)
        expected = (53, 67, 182, 196)
        self.assertIsNotNone(result)

    def test_mtcnn(self):
        detect = Detector()
        img = "/home/ashwin/Vision/test/assets/face.jpeg"
        set_detector = detect.set_detector(detector="mtcnn")
        result = detect.detect_face(img_path=img)
        expected = (51, 44, 123, 164)
        self.assertIsNotNone(result)

    def test_haar(self):
        detect = Detector()
        img = "/home/ashwin/Vision/test/assets/face.jpeg"
        set_detector = detect.set_detector(detector="haar")
        result = detect.detect_face(img_path=img)
        expected = (33, 42, 193, 202)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
