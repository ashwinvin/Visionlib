import unittest
from visionlib.face.detection import FDetector
from visionlib.gender.detection import GDetector

class TestDetection(unittest.TestCase):
    def test_hog(self):
        detect = FDetector()
        img = "/home/ashwin/Vision/test/assets/face.jpeg"
        img2 = "/home/ashwin/Vision/test/assets/man.jpg"
        set_detector = detect.set_detector(detector="hog")
        result = detect.detect_face(img_path=img, show=True)
        result2 = detect.detect_face(img_path=img2, show=True)
        expected = (53, 67, 182, 196)
        self.assertIsNotNone(result)

    def test_mtcnn(self):
        detect = FDetector()
        img = "/home/ashwin/Vision/test/assets/face.jpeg"
        img2 = "/home/ashwin/Vision/test/assets/man.jpg"
        set_detector = detect.set_detector(detector="mtcnn")
        result = detect.detect_face(img_path=img, show=True)
        result2 = detect.detect_face(img_path=img2, show=True)
        expected = (51, 44, 123, 164)
        self.assertIsNotNone(result)

    def test_haar(self):
        detect = FDetector()
        img = "/home/ashwin/Vision/test/assets/face.jpeg"
        img2 = "/home/ashwin/Vision/test/assets/man.jpg"
        set_detector = detect.set_detector(detector="haar")
        result = detect.detect_face(img_path=img, show=True)
        result2 = detect.detect_face(img_path=img2, show=True)
        expected = (33, 42, 193, 202)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
