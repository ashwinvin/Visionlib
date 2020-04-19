import dlib
from ..utils.imgutils import Image


class Hog_detector:
    def __init__(self):
        self.image_util = Image()
        self._dlib_hog_model = dlib.get_frontal_face_detector()

    def detect(self, img):
        """
        Detect faces using dlib's hog based detector.

        Args:
            img_path: Path to the image for detection

        Returns:
            A tuple containg 4 lists which correspond
            to x, y, w, h of bounding box respectively.

        """
        d_img = img
        if d_img is None:
            raise Exception("Provided Path {0} is invaild ".format(img))
        else:
            d_img = self.image_util.resize(d_img, 600, 300)
            detected_img = self._dlib_hog_model(d_img, 1)
            box_lst = []
            for face in detected_img:
                x = face.left()
                y = face.top()
                w = face.right()
                h = face.bottom()
                box_lst.append([x, y, w, h])
            return box_lst
