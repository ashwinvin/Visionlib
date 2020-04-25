from visionlib.face.detection import FDetector
from visionlib.gender.detection import GDetector
from visionlib.utils.imgutils import Image
import cv2
import argparse

# Configre the parser.
parser = argparse.ArgumentParser()
parser.add_argument("vid_path", help="Path to image")
args = parser.parse_args()

# Instantiating the required classes.
detector = FDetector()
Gdetector = GDetector()
im_utils = Image()

detector.set_detector("dnn")
# Read the video
detection = detector.vdetect_face(args.vid_path)
run = True
while run:
    # Get frames and bounding boxes of faces
    for img, boxes in detection:
        for box in boxes:
            # Get the face by cropping
            c_img = im_utils.crop(img, box)
            # Apply Gender Detection
            gender = Gdetector.detect_gender(c_img)
            # format the label
            label = "{}: {:.2f}%".format(gender[0], gender[1] * 100)
            # Put padding for rendering the label
            Y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 10
            cv2.putText(img, label, (box[0], Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("pic2", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            run = False
