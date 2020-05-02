from visionlib.object.classifier.detection import CDetector
import argparse
import cv2

# Configure the parser
parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="Path to image")
parser.add_argument("--model",
                    help="Model to use for detecting",
                    dest="model",
                    default="inception")
args = parser.parse_args()

detector = CDetector()

# Read the image
img = cv2.imread(args.img_path)

detector.set_detector(args.model)

predictions = detector.predict(img)

for prediction in predictions:
    print("Detected {0} with confidence {1}".format(prediction[0],
                                                    prediction[1]))
