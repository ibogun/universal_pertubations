"""
Unit tests.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import deep_fool
import numpy as np

class DeepFoolTest(tf.test.TestCase):
    """Test suite for deep fool."""

    def setUp(self):
        """Setup."""

        checkpoint_dir = "../checkpoints/"
        checkpoint_file = "inception_resnet_v2_2016_08_30.ckpt"
        path = os.path.join(checkpoint_dir, checkpoint_file)

        self._image_path = '../images/n02099601_10.JPEG'
        self._model = deep_fool.InceptionImageNetClassifier(path)

    def test_size(self):
        """Tests the size of the model."""
        with self.test_session():
            # create empty image given size.
            input_size = self._model.input_dims()

            # Tensor is expected to be image tensor
            self.assertEqual(len(input_size), 3)
            self.assertGreater(input_size[0], 0)
            self.assertGreater(input_size[1], 0)

    def test_read_and_preprocess(self):
        """Tests if image can be read and preprocessed."""
        image = self._model.read_and_preprocess(self._image_path)
        self.assertEqual(np.shape(image), self._model.input_dims())


    def test_inference(self):
        """Tests if the inference can be performed."""
        image = self._model.read_and_preprocess(self._image_path)
        probs = self._model(image)
        self.assertEqual(len(probs), self._model.num_classes())



if __name__ == '__main__':
    tf.test.main()
