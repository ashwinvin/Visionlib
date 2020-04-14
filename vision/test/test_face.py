import unittest
from ..face.detection import FaceDetection


class TestDetection(unittest.TestCase):
    def test_dlib(self):
        detect = FaceDetection()
        img = "/home/ashwin/Vision/vision/test/assets/face.jpeg"
        result = detect.dlib_hog_detector(img_path=img)
        expected = ([53], [67], [182], [196])
        self.assertIsNotNone(result)
        self.assertTupleEqual(result, expected)

    def test_mtcnn(self):
        detect = FaceDetection()
        img = "/home/ashwin/Vision/vision/test/assets/face.jpeg"
        result = detect.mtcnn_detector(img_path=img)
        expected = ([51], [44], [123], [164])
        self.assertIsNotNone(result)
        self.assertTupleEqual(result, expected)

    def test_haar(self):
        detect = FaceDetection()
        img = "/home/ashwin/Vision/vision/test/assets/face.jpeg"
        result = detect.haar_detector(img_path=img)
        expected = ([33], [42], [193], [202])
        self.assertIsNotNone(result)
        self.assertTupleEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
