import os
import cv2
import numpy as np


class Image:
    '''
    This class contains all the image utils for the package
    but You can also use it.

    Methods:
        bgr_to_grey
        bgr_to_rgb
        resize
        crop
        read_img
        read_video
    '''
    def __init__(self):
        pass

    def bgr_to_grey(self, frame):
        """
        Converts image to greyscale.

        Args:
            frame: numpy array
                    frame should be a numpy like array.
        Returns:
            c_frame: numpy array
                    converted image
        """
        c_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return c_frame

    def bgr_to_rgb(self, frame):
        """
        Converts image to rgb format.

        Args:
            frame: numpy array
                    frame should be a numpy like array.
        Returns:
            c_frame: numpy array
                    converted image
        """
        c_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return c_frame

    def resize(self, frame, x, y):
        """
        Resizes the image.

        Args:
            frame: numpy array
                    frame should be a numpy like array.
        Returns:
            r_frame: numpy array
                    Resized image
        """
        r_frame = cv2.resize(frame, dsize=(x, y), interpolation=cv2.INTER_AREA)
        return r_frame

    def crop(self, img, cords):
        """
        Crops the image.

        Args:
            frame: numpy array
                    frame should be a numpy like array.
        Returns:
                Cropped image
        """
        crop_image = img[cords[1]:cords[3], cords[0]:cords[2]]
        return crop_image

    def read_img(self, img_path):
        """
        Reads the image from path.

        Args:
           img_path: str
                    Takes an absolute path to the image.
        Returns:
            frame: numpy array
                    Returns an instance of the image if the
                    path given is correct else None.
        """
        file_exists = os.path.exists(img_path)
        if file_exists is True:
            self.frame = cv2.imread(img_path)
            return self.frame
        else:
            return None

    def read_video(self, video_path):
        """
        Reads the video from path.

        Args:
           video_path: str
                    Takes an absolute path to the video.
                    Set video path to 0 for webcam.
        Returns:
            frame: numpy array
                    Returns an instance of the video if the
                    path given is correct else None.
        """
        file_exists = os.path.exists(video_path)
        if file_exists is True:
            video = cv2.VideoCapture(video_path)
            return video
        else:
            return None
