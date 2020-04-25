from visionlib.face.detection import FDetector
import cv2
import argparse

# Configre the parser.
parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="Path to image")
args = parser.parse_args()

# Instantiating the required classes.
detector = FDetector()

# Read the image
img = cv2.imread(args.img_path)
detector.set_detector("hog")
# Apply face detection and show image
d_img, boxes = detector.detect_face(img, show=True)
for box in boxes:
    print(box)
