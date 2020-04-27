import dlib
from ..utils.imgutils import Image
import logging

class Hog_detector:
    def __init__(self):
        self.image_util = Image()
        self._dlib_hog_model = dlib.get_frontal_face_detector()

    def detect(self, img=None, enable_gpu=False):
        """
        Detect faces using dlibs hog based detector.

        Args:
            img_path: Path to the image for detection

        Returns:
                box_lst : A list of list
                    Each element in list correspond to
                    x, y, w, h of bounding box respectively.
        """
        d_img = img
        if d_img is None:
            raise Exception("No image received ".format(img))
        else:
            if enable_gpu:
                logging.warning("GPU is not supported for hog detector")
            detected_img = self._dlib_hog_model(d_img, 1)
            box_lst = []
            confidences = []
            for face in detected_img:
                x = face.left()
                y = face.top()
                w = face.right()
                h = face.bottom()
                box_lst.append([x, y, w, h])
                confidences.append(None)
            return box_lst, confidences
