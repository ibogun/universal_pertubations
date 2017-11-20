"""
Unit tests.
"""

import tensorflow as tf
import deep_fool

class DeepFoolTest(tf.test.TestCase):
    """Test suite for deep fool."""

    def setUp(self):
        """Setup."""
        self._model = deep_fool.MulticlassClassifier()

    def test_size(self):
        """Tests the size of the model."""
        with self.test_session():
            # create empty image given size.
            input_size = self._model.input_dims()

            # Tensor is expected to be image tensor
            self.assertEqual(input_size.size(), 2)
            self.assertGreater(input_size[0], 0)
            self.assertGreater(input_size[1], 0)

    def test_inference(self):
        """Tests if the inference can be performed."""
        with self.test_session():
            input_size = self._model.input_dims()
            # Create a tensor of the given shape.
            zeros = tf.zeros(input_size, tf.float32)
            output_size = self._model.num_classes()

            output = self._model(zeros)
            self.assertEquals(output.size(), output_size)



if __name__ == '__main__':
    tf.test.main()
