from visionlib.object.detection.detection import ODetection
import argparse
import numpy as np
import cv2

# Instantiating the required classes.
detector = ODetection()
# Configre the parser.
parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="Path to image")
parser.add_argument("--model-path", help="Path to custom model", dest="model_path")
parser.add_argument("--model", help="The model to use for detection", dest="model")
parser.add_argument("--label-path", help="Path to custom labels", dest="label_path")
parser.add_argument(
    "--save-path", help="Path to save the extracted objects", dest="save_path"
)
parser.add_argument(
    "--cfg-path", help="Path to custom model's config file", dest="cfg_path"
)
parser.add_argument(
    "--enable-gpu", help="Set to true to enable gpu support",
    dest="enable_gpu", default=False, type=bool,
)

args = parser.parse_args()
# Read the image
img = cv2.imread(args.img_path)

# we creating this copy to use this for extract_objects function, so as
# to get clear images. If we simply assign this variable to img
# it will only create a reference to the variable and any change to it would also
# change the other variable.
sep_img = np.copy(img)

detector.set_detector(args.model)
# Detect the detect_objects
if args.cfg_path is not None or args.model_path is not None:
    detector.set_detector(
        model_path=args.model_path, cfg_path=args.cfg_path, label_path=args.label_path
    )
box, label, conf = detector.detect_objects(img, enable_gpu=args.enable_gpu)
print(box, label, conf)
# Draw boxes on image
dimg = detector.draw_bbox(img, box, label, conf)
# show image
cv2.imshow("Object Detection using Visionlib", dimg)
cv2.waitKey(0)
if args.save_path is not None:
    ex_img = detector.extract_objects(
        img=sep_img, boxes=box, sdir=args.save_path, labels=label
    )
    for e_img in ex_img:
        pass
