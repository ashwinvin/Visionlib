import numpy as np
import cv2
import os
import sys
import logging
from visionlib.utils.webutils import web

class ODetection:
    """This class contains all functions to detect objects from an image.
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
        np.random.seed(62)
        self.web_util = web()
        self.min_confindence = 0.5
        self.threshold = 0.3

    def set_detector(self, model_name='tiny_yolo', model_path=None, cfg_path=None,
                     label_path=None):
        '''Set's the detector to use. Can be tiny-yolo or yolo.
        Setting to tiny-yolo will use yolov3-tiny.
        Setting to yolo will use yolov3.

        Args:
            model_name (str):
                    The model to use. If the given model is
                    not present in pc, it will download and use it.
            model_path (str):
                        Set this to path where the custom model You want
                        to load is.
            cfg_path (str):
                        Set this to path where the config file for custom model,
                        You want to load is.
            label_path (str):
                        Set this to path where the labels file for custom model,
                        You want to load is.
        '''

        if model_path is not None and cfg_path is not None:
            if os.path.exists(model_path) & os.path.exists(cfg_path):
                model = model_path
                cfg = cfg_path
                if label_path is not None:
                    labels = label_path
                else:
                    raise Exception("Provided label path does not exist")
            else:
                raise Exception("Provided model path or config path does not exist")

        elif model_name == "Tiny-yolov3":
            model, cfg, labels = self.web_util.download_file(model_name, labels=True)

        elif model_name == "Yolov3":
            model, cfg, labels = self.web_util.download_file(model_name, labels=True)

        else:
            raise Exception("Invalid model name")

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
        '''This method is used to detect objects in an image.

        Args:
            frame (np.array):
                        Image to detect objects from.

            enable_gpu (bool):
                         Set to true if You want to use gpu.

        Returns:
            list :
                The detected bounding box.

            list :
                The detected class.

            list :
                Confidence for each detected class

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
        '''Draw's Box around the detected objects.

        Args
            img (numpy.array):
                The image to draw bounding boxes
            bbox (list):
                bounding boxes given detect_objects function.
            labels (list):
                labels given detect_objects function.
            confidence (list):
                Confidence for the detected label.

        Returns
            numpy.array :
                The image with bounding boxes and labels.
        '''

        for i, label in enumerate(labels):
            color = self.colours[self.labels.index(label)]
            color = [int(x) for x in color]
            conf = round(confidence[i] * 100, 2)
            label = label + " " + str(conf)
            cv2.rectangle(img, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color, 2)
            cv2.putText(img, label, (bbox[i][0], bbox[i][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img
