from visionlib.face.detection import FDetector
from visionlib.utils.webutils import web
import cv2
import argparse

# Configre the parser.
parser = argparse.ArgumentParser()
parser.add_argument("vid_path", help="Path to image")
parser.add_argument(
    "--enable-gpu",
    help="Set to true to enable gpu support",
    dest="enable_gpu",
    default=False,
    type=bool,
)
args = parser.parse_args()

# Instantiating the required classes.
detector = FDetector()
web_util = web()

detector.set_detector("dnn")
# Read the video and apply face detection.
detection = detector.vdetect_face(
    vid_path=args.vid_path, show=True, enable_gpu=args.enable_gpu
)
for img, box, conf in detection:
    print(box, conf)
