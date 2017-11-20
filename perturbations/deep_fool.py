"""
Implementation of the Deep fool paper.
"""
import tensorflow as tf


class MulticlassClassifier(object):
    """Multiclass image classifier.
    Multiclass classifier is a function from
    R^{n x m} -> R^{k} where n x m is the size of the image and
    k is the number of classes.
    """

    def __call__(self, image):
        pass

    def grad(self, image):
        """
        Calculate the gradient for a given image
            :param image: mage used to calculate
        """
        pass

    def input_dims(self):
        """Returns input dimesions."""
        pass

    def num_classes(self):
        """Returns output number of classes."""
        pass

class DeepFool(object):
    """
    Implementation of the deepfool classifier
    """
    
    def __init__(self, model):
        self._model = model

    def fool(self, image):
        pass

