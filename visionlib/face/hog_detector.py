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

        Args
            img (numpy array)
                The image for detection

            enable_gpu (bool)
                Set to true to use gp support

        Returns
                box_lst (A list of list)
                    Each element in list correspond to
                    x, y, w, h of bounding box respectively.
        """
        d_img = img
        if d_img is None:
            raise Exception("No image received ".format(img))
        else:
            if enable_gpu:
                if dlib.cuda.get_num_devices() != 0:
                    dlib.DLIB_USE_CUDA = True
                else:
                    logging.fatal("No gpu detected")
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
            return [None, None] if (None in box_lst) else [box_lst, confidences]
