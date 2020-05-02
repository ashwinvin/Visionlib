import cv2
import os
import numpy as np
import logging
from pkg_resources import resource_filename, Requirement


class DnnDetector:
    def __init__(self):
        self.model = resource_filename(
            Requirement.parse("visionlib"), "visionlib" + os.path.sep
            + "models" + os.path.sep + "res10_300x300_ssd_iter_140000.caffemodel"
        )
        self.proto = resource_filename(
            Requirement.parse("visionlib"),
            "visionlib" + os.path.sep + "models" + os.path.sep + "deploy.prototxt"
        )
        self.model = cv2.dnn.readNetFromCaffe(self.proto, self.model)

    def detect(self, img=None, enable_gpu=False, threshold=0.5):
        """
        Detect faces using deep neural network.
        Uses res10_300x300_ssd_iter_140000.

        Args:
            img (nmpy array)
                Image for detection

            enable_gpu (bool)
                     Set to True if You want to use gpu for prediction.

            threshold (int)
                Min confidence for selecting the face.

        Returns
                box_lst (A list of list)
                    Each element in list correspond to
                    x, y, w, h of bounding box respectively.
        """
        if img is not None:
            if enable_gpu:
                logging.info("Activating gpu")
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            (h, w) = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.model.setInput(blob)

            detections = self.model.forward()
            box_lst = []
            confidences = []

            for i in range(0, detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf < threshold:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                box_lst.append([startX, startY, endX, endY])
                confidences.append(conf)

            return [None, None] if (None in box_lst) else [box_lst, confidences]
        else:
            raise Exception("No image received ".format(img))
