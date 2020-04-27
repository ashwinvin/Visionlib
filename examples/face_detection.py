from visionlib.face.detection import FDetector
import cv2
import argparse

# Configre the parser.
parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="Path to image")
parser.add_argument("--enable-gpu", help="Set to true to enable gpu support",
                    dest="enable_gpu", default=False, type=bool)
args = parser.parse_args()

# Instantiating the required classes.
detector = FDetector()

# Read the image
img = cv2.imread(args.img_path)
detector.set_detector("hog")
# Apply face detection and show image
d_img, boxes, conf = detector.detect_face(img, show=True, enable_gpu=args.enable_gpu)
for box in boxes:
    # print(conf)
    print(box)
