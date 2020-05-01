from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.applications.inception_v3 import InceptionV3
import cv2


class Inception:
    """This class is used to classify images
        using pre-trained Inception_v3 model.

        Methods:

            predict()
                Used to predict the classes from the image.
    """
    def __init__(self):
        self.model = InceptionV3(weights="imagenet")

    def __process_image(self, img):
        p_img = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)
        p_img = img_to_array(p_img)
        p_img = p_img.reshape((-1, p_img.shape[0], p_img.shape[1], p_img.shape[2]))
        p_img = preprocess_input(p_img)
        return p_img

    def predict(self, img=None, top=10):
        """This function is used to predict the possible classes in
            the image.

            Args
                img (numpy array)
                    The image for detecting classes

                top (int)
                    The top n classes detected by the model
                    to return

            Returns
                list
                    Returns a list of tuples with first value
                    is the detected class and second value is
                    the confidence in each tuple.
                    eg: [('coffee_mug', 88.80746364593506), ('cup', 9.322341531515121)]

            Raises
                AssertionError
                    Raised when no image is given
        """
        if img is not None:
            p_img = self.__process_image(img)
            predictions = self.model.predict(p_img)
            prediction_labels = decode_predictions(predictions)
            final_labels = []
            for i, label in enumerate(prediction_labels[0]):
                if i < top + 1:
                    final_labels.append((label[1], label[2] * 100))
                else:
                    break
            return final_labels
        else:
            raise AssertionError("No image provided")
