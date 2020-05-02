from mtcnn import MTCNN
from ..utils.imgutils import Image
import logging

class MTCNNDetector(object):
    def __init__(self):
        self.image_util = Image()
        self._mtcnn_model = MTCNN()

    def detect(self, img=None, enable_gpu=False):
        """
        Detect faces using mtcnn based detector.

        Args
            img (numpy array)
                The image for detection

        Returns
                box_lst : A list of list
                    Each element in list correspond to
                    x, y, w, h of bounding box respectively.
                    eg:
        [{'box': [285, 89, 175, 218], 'confidence': 0.996926486492157,
        """
        m_img = img
        if m_img is None:
            raise Exception("Provided Path {0} is invaild ".format(img))
        else:
            if enable_gpu:
                logging.warning("GPU is not supported for MTCNN")
            detected_img = self._mtcnn_model.detect_faces(m_img)
            box_lst = []
            confidences = []
            for face in detected_img:
                x = face["box"][0]
                y = face["box"][1]
                w = face["box"][2]
                h = face["box"][3]
                box_lst.append([x, y, w + x, h + y])
                conf = face['confidence']
                confidences.append(conf)
            return [None, None] if (None in box_lst) else [box_lst, confidences]
