import unittest
from cs3bMain import NNData as Nnd
from cs3bMain import DataMismatchError as DataME


class TestNNData(unittest.TestCase):
    """T.D.D for the NNData Class, testing the following:
    1. NNData.load_data() raises DataMismatchError if features and labels
       have different lengths when calling and verify that self._features and
       self._labels are set to None.
    2. NNData.load_data() raises ValueError if features or labels contain non-float
       values, & that they are set to none.
    3. Verify that if invalid lists are passed to the constructor e.g lists of
       of different length or non-homogeneous, sets self._labels & self._features to None
    4. Verify that NNData limits the training factor to 0 if -ve number is passed.
    5. Verify that NNData limits training factor to one if num greater than 1 is passed.
    """

    def setUp(self):
        self.nndata = Nnd(train_factor=.9, labels=None, features=None)

    # def test_lists_length_equality(self):
    #     features = [2, 4, 6, 7]
    #     labels = [2, 3, 5]
    #     with self.assertRaises(DataME):
    #         self.nndata.load_data(features=features, labels=labels)

    def test_return_None_with_unequal_lists(self):
        features = [2, 4, 6, 7]
        labels = [2, 3, 5]
        self.assertIsNone(self.nndata.load_data(features=features, labels=labels), msg="The Lists are not equal!")


if __name__ == "__main__":
    unittest.main()
