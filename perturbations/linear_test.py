import linear
import unittest


class LinearPertubationsTest(unittest.TestCase):
    def testBasic(self):
        print(3)
        self.assertEqual(linear.sample_function(3), 3)


if __name__ == '__main__':
    unittest.main()
  