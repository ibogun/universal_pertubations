"""
Implementation of the Deep fool paper.
"""
import tensorflow as tf
from inception_preprocessing import preprocess_image
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

slim = tf.contrib.slim

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

class InceptionImageNetClassifier(MulticlassClassifier):

    def __init__(self, path_to_checkpoint):
        """Creates an instance of the classifier."""
        self._checkpoint_path = path_to_checkpoint
        self._input_dims = (299, 299, 3)
        # Label 0 is for background.
        self._num_classes = 1001

    def input_dims(self):
        """Returns input size as a tuple"""
        return self._input_dims

    def num_classes(self):
        """Returns output number of classes as a scalar."""
        return self._num_classes

    def read_and_preprocess(self, image_path):
        """Reads and preprocesses the image from a path."""

        with tf.Graph().as_default():
            image_encoded = tf.gfile.FastGFile(image_path, 'rb').read()
            image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)

            input_dims = self.input_dims()
            processed_image_tensor = preprocess_image(image_decoded, input_dims[0],
                                                      input_dims[1], is_training=False)

            with tf.Session() as sess:
                processed_image = sess.run([processed_image_tensor])
                return processed_image[0]


    def __call__(self, image):
        """Image is assumed to be RBD image."""

        with tf.Graph().as_default():
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                processed_image = tf.placeholder(tf.float32, shape = self.input_dims())
                processed_images = tf.expand_dims(processed_image, 0)

                # Note(bogun): The class 0 is used to represent "non of the above" class also known as background.
                logits, _ = inception_resnet_v2(processed_images, num_classes=self.num_classes(), is_training=False)
                probabilities = tf.nn.softmax(logits)

                init_fn = slim.assign_from_checkpoint_fn(self._checkpoint_path, slim.get_model_variables('InceptionResnetV2'))

                with tf.Session() as sess:
                    init_fn(sess)
                    probabilities = sess.run([probabilities], feed_dict = {processed_image : image})

                    probs = probabilities[0]
                    return probs[0]
    


class DeepFool(object):
    """
    Implementation of the deepfool classifier
    """
    
    def __init__(self, model):
        self._model = model

    def fool(self, image):
        pass

