from visionlib.object.classifier.inception_detector import Inception
from visionlib.object.classifier.xception_detector import Xceptionv1
from visionlib.object.classifier.vgg_detector import VGG

class CDetector:
    def __init__(self):
        self.detector = Inception()

    def set_detector(self, detector):
        """This method is used to set detector to be used to predict the
        classe in an image.

        Args:
            detector (str) :
                The detector to be used. Can be any of the following:
                inception, xception, vgg. inception will be used as default.
        """
        if detector == 'inception':
            self.detector = self.detector
        elif detector == 'xception':
            self.detector = Xceptionv1()
        elif detector == 'vgg':
            self.detector = VGG()
        else:
            raise AssertionError('Invalid Selection')

    def predict(self, img=None, top=10):
        if img is not None:
            predictions = self.detector.predict(img=img, top=top)
            return predictions
        else:
            raise AssertionError('No image is given')
