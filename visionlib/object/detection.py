import numpy as np
import cv2
import os
import sys
from visionlib.utils.webutils import web

class Detection:
    """
    This class contains all functions to detect objects from an image.
                . . .

    Methods:

        detect_objects():

            Used to detect objects from an image. Returns the bounding
            boxes, labels and confidence. Uses detector set by set_detector() method.

        draw_box():

            Used to draw the bounding box, labels and confidence in an image. Returns
            the frame with bounding boxes. Uses detector set by set_detector() method.

        set_detector():

            Used to set detector to used by detect_objects() method. If not set
            will use tiny yolo as default.

    """

    def __init__(self):
        self.model = None
        self.model_ln = None
        np.random.seed(6865)
        self.web_util = web()
        self.min_confindence = 0.5
        self.threshold = 0.3

    def set_detector(self, model_name='tiny_yolo'):
        '''
        Set's the detector to use. Can be tiny-yolo or yolo.
        Setting to tiny-yolo will use yolov3-tiny.
        Setting to yolo will use yolov3.

        Args:
            model_name(str):The model to use. If the given model is
                not present in pc, it will download and use it.
        '''
        if model_name == "tiny_yolo":
            model_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
            model_file_name = 'yolov3-tiny.weights'
            cfg_url = "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg"
            cfg_file_name = 'yolov3-tiny.cfg'

        elif model_name == "yolo":
            model_url = "https://pjreddie.com/media/files/yolov3.weights"
            model_file_name = 'yolov3.weights'
            cfg_url = 'https://github.com/arunponnusamy/object-detection-opencv/raw/master/yolov3.cfg'
            cfg_file_name = "yolov3.cfg"
        else:
            raise Exception("Invalid Selection")

        labels_url = 'https://github.com/arunponnusamy/object-detection-opencv/raw/master/yolov3.txt'
        labels_file_name = 'yolo_classes.txt'

        labels = self.web_util.download_file(labels_url, labels_file_name)
        model = self.web_util.download_file(model_url, model_file_name)
        cfg = self.web_util.download_file(cfg_url, cfg_file_name)

        if model and cfg is not None:

            with open(labels, 'r') as file:
                self.labels = file.read().strip().split("\n")
            self.colours = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

            self.model = cv2.dnn.readNetFromDarknet(cfg, model)
            self.model_ln = self.model.getLayerNames()
            self.model_ln = [
                self.model_ln[i[0] - 1] for i in self.model.getUnconnectedOutLayers()
            ]

    def detect_objects(self, frame, enable_gpu=False):
        '''
        This method is used to detect objects in an image.

        Args:
            img : cv2.imshow return output
                This argument must the output which similar to
                opencv's imread method's output.
            enable_gpu : bool
                Set to true if You want to use gpu.
        Returns:
            bbox : The detected bounding box.
            labels : The deteted class.
            confidence : Confidence for each detected class
        '''
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )

        if enable_gpu:
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.model.setInput(blob)
        layerOutputs = self.model.forward(self.model_ln)
        boxes, confidences, classIDs = [], [], []

        for output in layerOutputs:

            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.min_confindence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, self.min_confindence, self.threshold
        )
        bbox = []
        label = []
        conf = []
        if len(idxs) > 0:
            for ix in idxs:
                ix = ix[0]
                box = boxes[ix]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                bbox.append([int(x), int(y), int(x + w), int(y + h)])
                label.append(str(self.labels[classIDs[ix]]))
                conf.append(confidences[ix])
        return bbox, label, conf

    def draw_bbox(self, img, bbox, labels, confidence):
        '''
        Draw's Box around the detected objects.

        Args
            img : The image to draw
            bbox : bounding boxes given detect_objects function.
            labels : labels given detect_objects function.
            confidence : Confidence for the detected label.
        Returns
            img : The image with bounding boxes and labels.
        '''

        for i, label in enumerate(labels):
            color = self.colours[self.labels.index(label)]
            color = [int(x) for x in color]

            cv2.rectangle(img, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color, 2)
            cv2.putText(img, label, (bbox[i][0], bbox[i][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img
