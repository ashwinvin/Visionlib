from visionlib.object.detection import Detection
import argparse
import cv2

# Instantiating the required classes.
detector = Detection()
detector.set_detector("tiny_yolo")
# Configre the parser.
parser = argparse.ArgumentParser()
parser.add_argument("vid_path", help="Path to image")
args = parser.parse_args()
# Read the Video
vid = cv2.VideoCapture(args.vid_path)
run = True
while run:
    # Read the frame
    status, img = vid.read()
    # Check if video ended
    if not status:
        break
    else:
        # Detect the objects.
        box, label, conf = detector.detect_objects(img)
        d_img = detector.draw_bbox(img, box, label, conf)
        # Check if there is objects
        if d_img is not None:
            # Show the object
            cv2.imshow("Object Detection", d_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                run = False
        # if no object is found show default image
        else:
            cv2.imshow("Object Detection ", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                run = False
