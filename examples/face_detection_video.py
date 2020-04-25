from visionlib.face.detection import FDetector
import cv2
import argparse

# Configre the parser.
parser = argparse.ArgumentParser()
parser.add_argument("vid_path", help="Path to image")
args = parser.parse_args()

# Instantiating the required classes.
detector = FDetector()

detector.set_detector("hog")
# Read the video and apply face detection.
detection = detector.vdetect_face(args.vid_path, show=True)
for img, box in detection:
    print(box)
