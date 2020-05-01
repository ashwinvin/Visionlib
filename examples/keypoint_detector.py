from visionlib.face.detection import FDetector
from visionlib.keypoints.detection import KDetector
import argparse
import cv2

# Configuring the parser
parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="Path to image")
parser.add_argument(
    "--key-model",
    help="Model to use for detecting keypoints",
    dest="key_model",
    default="dlib",
)
parser.add_argument(
    "--face-model",
    help="Model to use for detecting face",
    dest="face_model",
    default="hog",
)
parser.add_argument(
    "--enable-gpu",
    help="Set to true to enable gpu support",
    dest="enable_gpu",
    default=False,
    type=bool,
)
args = parser.parse_args()

# Instantiating the required class
fdetector = FDetector()
kdetector = KDetector()

# Read the image
img = cv2.imread(args.img_path)

# Set the detectors
kdetector.set_detector(args.key_model)
fdetector.set_detector(args.face_model)

# Detect face and the points
fimg, boxes, conf = fdetector.detect_face(img, enable_gpu=True)
points = kdetector.detect_keypoints(img, rects=boxes)

# Draw the points
pimg = kdetector.draw_points(fimg, points, show=True)
