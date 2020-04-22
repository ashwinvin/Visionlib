import cv2
from ..utils.imgutils import Image


class HaarDetector:
    def __init__(self):
        self._haar_cascades = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.image_util = Image()

    def detect(self, img):
        """
        Detect faces using opencv's haar based detector.

        Args:
            img_path: Path to the image for detection

        Returns:
                box_lst : A list of list
                    Each element in list correspond to
                    x, y, w, h of bounding box respectively.
        """
        h_img = img
        if h_img is None:
            raise Exception("Provided Path {0} is invaild ".format(img))
        else:
            grey_img = self.image_util.bgr_to_grey(h_img)
            detected_img = self._haar_cascades.detectMultiScale(grey_img, 1.3, 5)
            box_lst = []
            for (x, y, w, h) in detected_img:
                box_lst.append([x, y, w + x, h + y])
                return box_lst
