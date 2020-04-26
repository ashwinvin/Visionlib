# Visionlib

A simple high level api made for assisting in cv-related projects.

## Features

- Track faces using
  - MTCNN module
  - Dlib hog Based detector
  - Opencv Haar casscades
  - Dnn based model
- Predict Gender
- Detect Objects
  - Yolo v3
  - tiny-yolo

### Installation

#### Dependencies

`sudo apt-get install build-essential cmake pkg-config`

`sudo apt-get install libx11-dev libatlas-base-dev`

`sudo apt-get install libgtk-3-dev libboost-python-dev`

This should install Dependencies required by dlib.

`pip install visionlib`

This will install visionlib.

##### Optional

If You want to install from source
`git clone https://github.com/ashwinvin/Visionlib.git`

`cd visionlib`

`pip install .`

### Face Detection

Detecting face in an image is easy . This will return the image with bounding box and box coordinates.

`from visionlib.face.detection import FDetector`

`detector = FDetector()`

`detector.detect_face(img, show=True)`

This would detect face and display it automatically.

`detector.set_detector("mtcnn")`
Dont like the default detector?, change it like this.

#### Examples

![Detection](docs/images/face_detected.jpg)

![Detection](docs/images/face_detected_group.jpg)

### Gender Detection

Once face is detected, it can be passed on to detect_gender() function to recognize gender. It will return the labels (man, woman) and associated probabilities.Like this

`from visionlib.gender.detection import GDetector`

`Gdetector = GDetector()`

`pred, confidence = Gdetector.detect_gender(c_img)`

##### Example

![Gender Detection](docs/images/gender_detected_single.jpg)

### Object Detection

Detecting common objects in the scene is enabled through a single function call detect_objects(). It will return the labeled image for the detected objects in the image. By default it uses yolov3-tiny model.
`from visionlib.object.detection import Detection`

`import cv2`

`detector = Detection()`

`d_img = detector.detect_objects(img)`

You can enable gpu usage by 
`detector.detect_objects(img, enable_gpu=True)`

#### Example
![object Detection](docs/images/object_detected_objects.jpg)


For more examples look in examples directory
