from visionlib.object.classifier.xception_detector import Xceptionv1
from visionlib.object.classifier.inception_detector import Inception
import argparse
import cv2

# Configure the parser
parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="Path to image")
parser.add_argument(
    "--model", help="Model to use for detecting", dest="model", default="inception"
)
args = parser.parse_args()

# Read the image
img = cv2.imread(args.img_path)

if args.model == 'xception':
    # Instantiating the required class
    xception = Xceptionv1()
    # Predicting classes
    xcpreds = xception.predict(img)
    print("Xeption Predictions")
    for pred in xcpreds:
        print(pred)
else:
    # Instantiating the required class
    inception = Inception()
    # Predicting classes
    inpreds = inception.predict(img)
    print("Inception Predictions")
    for pred in inpreds:
        print(pred)
