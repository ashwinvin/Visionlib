Welcome to Visionlib’s documentation!
*************************************


Face Detection
**************

class visionlib.face.detection.FDetector

   This class contains all functions to detect face in an image.
      …

   Methods:

      detect_face():

         Used to detect face in an image. Returns the image with
         bounding boxes. Uses detector set by set_detector() method.

      vdetect_face():

         Used to detect face in a video. Yields the frame with
         bounding boxes. Uses detector set by set_detector() method.

      set_detector():

         Used to set detector to used by detect_face() method. If not
         set will use dnn detector as default.

   detect_face(img=None, show=False, enable_gpu=False)

      This method is used to detect face in an image.

      Args:
         img (numpy array) :
            This argument must the output which similar to opencv’s
            imread method’s output.

         show (bool) :
            Set True to show image via cv2.imshow method.

      Returns:
         img (np.array) :
            Returns a numpy array of the image with bounding box.

         box (list) :
            Returns x, y, w, h coordinates of the detected face
            Returns an empty list if no face is detected.

         confidences (list) :
            Returns the associated confidences for the detected face.

   set_detector(detector='dnn')

      This method is used to set detector to be used to detect faces
      in an image.

      Args:
         detector (str) :
            The detector to be used. Can be any of the following:
            haar, hog, mtcnn, dnn. Dnn will be used as default.

   vdetect_face(vid_path=None, show=False, enable_gpu=False)

      This method is used to detect face in an video

      Args:
         vid_path (str):
            Absolute Path to the video file.

         show (bool):
            Set True to show image via cv2.imshow method.

      Yields:
         img (np.array) :
            Returns a numpy array of the image with bounding box. ONLY
            given when show is set to True

         box (list) :
            Returns x, y, w, h coordinates of the detected face
            Returns an empty list if no face is detected.

         confidences (list) :
            Returns the associated confidences for the detected face.


Gender Detection
****************

class visionlib.gender.detection.GDetector

   This class contains all functions to detect gender of a given face
      …

   Methods:

      detect_gender():

         Used to detect gender from an face. Returns Predicted Gender
         and confidence.

   detect_gender(img=None, enable_gpu=False)

      This method is used to detect gender from an image.

      Args:
         img (numpy array)
            This argument must the output which similar to opencv’s
            imread method’s output.

         enable_gpu (bool) :
            Set to True if You want to use gpu for prediction.

      Returns:
         str :
            Returns the predicted gender.

         int :
            Returns the confidence for the predicted gender.


Object Detection
****************

class visionlib.object.detection.Detection

   This class contains all functions to detect objects from an image.
      …

   Methods:

      detect_objects():

         Used to detect objects from an image. Returns the bounding
         boxes, labels and confidence. Uses detector set by
         set_detector() method.

      draw_box():

         Used to draw the bounding box, labels and confidence in an
         image. Returns the frame with bounding boxes. Uses detector
         set by set_detector() method.

      set_detector():

         Used to set detector to used by detect_objects() method. If
         not set will use tiny yolo as default.

   detect_objects(frame, enable_gpu=False)

      This method is used to detect objects in an image.

      Args:
         frame (np.array):
            Image to detect objects from.

         enable_gpu (bool):
            Set to true if You want to use gpu.

      Returns:
         list :
            The detected bounding box.

         list :
            The detected class.

         list :
            Confidence for each detected class

   draw_bbox(img, bbox, labels, confidence)

      Draw’s Box around the detected objects.

      Args
         img (numpy.array):
            The image to draw bounding boxes

         bbox (list):
            bounding boxes given detect_objects function.

         labels (list):
            labels given detect_objects function.

         confidence (list):
            Confidence for the detected label.

      Returns
         numpy.array :
            The image with bounding boxes and labels.

   set_detector(model_name='tiny_yolo', model_path=None, cfg_path=None, label_path=None)

      Set’s the detector to use. Can be tiny-yolo or yolo. Setting to
      tiny-yolo will use yolov3-tiny. Setting to yolo will use yolov3.

      Args:
         model_name (str):
            The model to use. If the given model is not present in pc,
            it will download and use it.

         model_path (str):
            Set this to path where the custom model You want to load
            is.

         cfg_path (str):
            Set this to path where the config file for custom model,
            You want to load is.

         label_path (str):
            Set this to path where the labels file for custom model,
            You want to load is.


Keypoint Detection
******************

class visionlib.keypoints.detection.KDetector

   This class contains functions for getting
      keypoints for a space

   Methods

      set_detector()
         Set the detector to detect keypoints

      detect_keypoints()
         Detect the keypoints in a face.

      draw_points()
         Draws the detected points in an image

   detect_keypoints(img, rects=None)

      This function is used to detect keypoints.

      Args
         img (numpy array)
            The image for detection.

         rects (list)
            The coordinates of face. (Not Needed for mtcnn detector)

      Returns
         list
            Returns detected keypoint for each face

         img
            Returned only when no face is there for detection.

   draw_points(img, points, show=False, color=0, 50, 255)

      This function is used to draw the detected points
         into the image.

      Args
         img (numpy array)
            The image for drawing.

         points (list)
            The coordinates of the keypoints

         show (bool)
            Set to true if you want to display the image

         color (tuple)
            The color to display the points in.

      Returns
         numpy array
            The image with keypoints marked

   set_detector(detector)

      This function is used to set the detector
         to detect keypoints.

      Args
         detector (str)
            Can be ‘dlib’ or ‘mtcnn’.

      Raises
         Invalid Selection
            Raised when the selection is invalid

