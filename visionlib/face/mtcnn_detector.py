from mtcnn import MTCNN
from ..utils.imgutils import Image


class MTCNNDetector(object):
    def __init__(self):
        self.image_util = Image()
        self._mtcnn_model = MTCNN()

    def detect(self, img):
        """
        Detect faces using mtcnn based detector.

        Args:
            img_path: Path to the image for detection

        Returns:
            A tuple containg 4 lists which correspond
            to x, y, w, h of bounding box respectively.

        """
        m_img = img
        if m_img is None:
            raise Exception("Provided Path {0} is invaild ".format(img))
        else:
            detected_img = self._mtcnn_model.detect_faces(m_img)
            box_lst = []
            for face in detected_img:
                x = face["box"][0]
                y = face["box"][1]
                w = face["box"][2]
                h = face["box"][3]
                box_lst.append([x, y, w, h])
            return box_lst
