from visionlib.object.detection import Detection
import argparse
import cv2

# Instantiating the required classes.
detector = Detection()
# Configre the parser.
parser = argparse.ArgumentParser()
parser.add_argument("vid_path", help="Path to image")
parser.add_argument("--model-path", help="Path to custom model", dest="model_path")
parser.add_argument("--cfg-path", help="Path to custom model's config file", dest="cfg_path")
parser.add_argument("--label-path", help="Path to custom labels", dest="label_path")
parser.add_argument("--enable-gpu", help="Set to true to enable gpu support",
                    dest="enable_gpu", default=False, type=bool)
args = parser.parse_args()

if args.cfg_path is None or args.model_path is None:
    detector.set_detector(model_name="yolo")
else:
    detector.set_detector(model_path=args.model_path, cfg_path=args.cfg_path, 
                          label_path=args.label_path)

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
        box, label, conf = detector.detect_objects(img, enable_gpu=args.enable_gpu)
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
