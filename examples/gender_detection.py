from visionlib.face.detection import FDetector
from visionlib.gender.detection import GDetector
from visionlib.utils.imgutils import Image
import cv2
import argparse

# Configre the parser.
parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="Path to image")
args = parser.parse_args()
# Instantiating the required classes.
Fdetector = FDetector()
Gdetector = GDetector()
im_utils = Image()
# Read the image
img = cv2.imread(args.img_path)
Fdetector.set_detector("hog")
# Detect the Faces
d_img, boxes, conf = Fdetector.detect_face(img)
for box in boxes:
    # Get the face by cropping
    c_img = im_utils.crop(d_img, box)
    # Apply Gender Detection
    gender = Gdetector.detect_gender(c_img)
    # format the label
    label = "{}: {:.2f}%".format(gender[0], gender[1] * 100)
    # Put padding for rendering the label
    Y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 10
    cv2.putText(
        d_img, label, (box[0], Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    cv2.imshow("pic2", img)
    cv2.waitKey(0)
