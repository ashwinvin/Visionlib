from visionlib.object.detection import Detection
import argparse
import cv2

# Instantiating the required classes.
detector = Detection()
detector.set_detector("tiny_yolo")
# Configre the parser.
parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="Path to image")
parser.add_argument("--model-path", help="Path to custom model", dest="model_path")
parser.add_argument("--cfg-path", help="Path to custom model's config file", dest="cfg_path")
parser.add_argument("--label-path", help="Path to custom labels", dest="label_path")
parser.add_argument("--enable-gpu", help="Set to true to enable gpu support",
                    dest="enable_gpu", default=False, type=bool)
args = parser.parse_args()
# Read the image
img = cv2.imread(args.img_path)
# Detect the detect_objects
if args.cfg_path is None or args.model_path is None:
    detector.set_detector(model_name="yolo")
else:
    detector.set_detector(model_path=args.model_path, cfg_path=args.cfg_path, 
                          label_path=args.label_path)
box, label, conf = detector.detect_objects(img, enable_gpu=args.enable_gpu)
print(box, label, conf)
# Draw boxes on image
dimg = detector.draw_bbox(img, box, label, conf)
# show image
cv2.imshow("pic2", dimg)
cv2.waitKey(0)
