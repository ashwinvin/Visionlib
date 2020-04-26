from visionlib.object.detection import Detection
import argparse
import cv2

# Instantiating the required classes.
detector = Detection()
detector.set_detector("tiny_yolo")
# Configre the parser.
parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="Path to image")
args = parser.parse_args()
# Read the image
img = cv2.imread(args.img_path)
# Detect the objects
box, label, conf = detector.detect_objects(img)
print(box, label, conf)
# Draw boxes on image
dimg = detector.draw_bbox(img, box, label, conf)
# show image
cv2.imshow("pic2", dimg)
cv2.waitKey(0)
